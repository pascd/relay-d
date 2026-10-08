import time

from PyQt5.QtCore import QTimer
from relay_d.utils.coloring_logger import logger
from relay_d.utils.tf_topic import TfTopicSubscriptionError

# Import utils
from ..config.yaml_parser import YamlParser
from ..containers.topic_containers import TopicContainers
from ..record.topic_subscribers import TopicSubscribers
from ..record.container_visualizer import ContainerVisualizer
from ..record.data_recorder import DataRecorder
from ..record.container_manager import ContainerManager
from ..review.demo_validator import DemoValidator, CheckStatus
from ..settings.config_manager import config
from ...utils.custom_message_box import CustomMessageBox

class RecordPage:
    def __init__(self, ui_instance, yaml_parser: YamlParser, settings_page=None):
        self.ui = ui_instance
        self.yaml_parser = yaml_parser
        self.settings_page = settings_page
        self.topic_containers = TopicContainers(ui_instance)

        # Initialize the TopicSubscribers
        # with the main ROS node
        self.topic_subscribers = TopicSubscribers()

        # Initialize visualization and recording
        self.container_visualizer = ContainerVisualizer()
        self.data_recorder = DataRecorder(self.yaml_parser)
        self.container_manager = ContainerManager(
            ui_instance,
            yaml_parser,
            self.topic_containers,
            self.topic_subscribers,
            self.container_visualizer,
        )

        # Recording state
        self.is_recording = False
        self.recording_start_time = None

        # Session-level demo tracking (mirrors AppAPI._session_h5_files)
        self.session_h5_files: list = []
        self.last_validation_result = None
        self._demo_validator = DemoValidator()

        # Setup all button connections
        self.setup_connections()
        self.config_signal_manager = None

        # Setup recording directory
        # self.data_recorder.setup_recording(settings_page=self.settings_page)

        # Timer for TF updates
        self.tf_update_timer = QTimer()
        self.tf_update_timer.timeout.connect(self._update_tf_lookups)

        # Connect to config changes
        config.settings_changed.connect(self._on_settings_changed)

    def _on_settings_changed(self, settings):
        """Handle settings changes via signal"""
        if "recording_frequency" in settings:
            freq = settings["recording_frequency"]
            if freq and freq > 0:
                self.data_recorder.recording_frequency = freq
                logger.info(f"Recording frequency updated to {freq} Hz")

        if "hdf5_save_folder" in settings:
            self.data_recorder.output_directory = settings["hdf5_save_folder"]
            logger.info(f"HDF5 save folder updated to: {settings['hdf5_save_folder']}")

        if "enable_compression" in settings:
            logger.info(f"Compression set to {settings['enable_compression']}")

        if "freeze_containers" in settings:
            logger.info(f"Freeze containers set to {settings['freeze_containers']}")

    def _update_tf_lookups(self):
        """Update TF lookups periodically"""
        self.topic_subscribers.update_tf_lookups()

    def connect_to_config_signals(self, config_signal_manager):
        """Connect to ConfigPage signals"""
        # Connect the signals to appropriate slots
        config_signal_manager.config_loaded.connect(self._on_config_loaded)
        config_signal_manager.config_cleared.connect(self._on_config_cleared)
        config_signal_manager.config_saved.connect(self._on_config_saved)

    def _on_config_saved(self, file_path):
        """Handle config saved signal - reload containers"""
        logger.info(f"Config saved, reloading containers from: {file_path}")

        # Reload the config file
        self.yaml_parser.load_yaml_file(file_path)

        # Recreate containers with new configuration
        self._create_containers()

    def _on_config_loaded(self, file_path):
        """Handle config loaded signal"""
        logger.info(f"Config loaded from: {file_path}")

        # Load config first
        self.yaml_parser.load_yaml_file(file_path)

        # Register the topic_subscribers node with the UI for spinning
        if hasattr(self.ui, "register_ros_node"):
            self.ui.register_ros_node(self.topic_subscribers)
            self.ui.register_ros_node(self.data_recorder)
            logger.info("Registered topic_subscribers and data_recorder nodes with UI")

        # Create the containers with visualization
        self._create_containers()

        # Start TF update timer at recording_frequency (not hardcoded 10 Hz)
        freq = self.data_recorder.recording_frequency if self.data_recorder else 20
        interval_ms = max(50, int(1000 / freq))   # floor at 50 ms (20 Hz max for UI)
        self.tf_update_timer.start(interval_ms)
        logger.info(f"TF update timer started at {freq} Hz ({interval_ms} ms)")

        ## Enable the buttons
        ## Buttons to start, record and pause
        if hasattr(self.ui, "btn_start_record"):
            self.ui.btn_start_record.setEnabled(True)

        if hasattr(self.ui, "btn_pause_record"):
            self.ui.btn_pause_record.setEnabled(False)

        if hasattr(self.ui, "btn_stop_record"):
            self.ui.btn_stop_record.setEnabled(False)

        ## Set grippers keys for state modification
        self.gripper_key_map = {}
        input_data = self.yaml_parser.get_input_data()

        for entry in input_data:
            if entry["name"].startswith("gripper") and "topic" in entry:
                logger.info(f"Found gripper entry: {entry['name']}")
                # For gripper, we now use key events - open/close keys should be defined separately
                # or handled through the GUI

        logger.info(f"Total gripper mappings: {len(self.gripper_key_map)}")

        # Update status
        self.update_config_status(f"Config loaded: {file_path}")

    def _on_config_cleared(self):
        """Handle config cleared signal"""
        logger.info("Config cleared")

        # Stop TF update timer
        self.tf_update_timer.stop()

        # Stop all visualizations
        self.container_visualizer.stop_all_visualizations()

        # Clear containers and subscribers
        self.topic_subscribers.clear_all_subscribers()
        self.topic_containers.clear_all_containers()

        # RESET layout references
        if hasattr(self.ui, "page_start"):
            self.topic_containers.reset_layout_references(self.ui.page_start)

        ## Enable the buttons
        ## Buttons to start, record and pause
        if hasattr(self.ui, "btn_start_record"):
            self.ui.btn_start_record.setEnabled(False)

        if hasattr(self.ui, "btn_pause_record"):
            self.ui.btn_pause_record.setEnabled(False)

        if hasattr(self.ui, "btn_stop_record"):
            self.ui.btn_stop_record.setEnabled(False)

        # Update status
        self.update_config_status("No config loaded")

    def _create_containers(self):
        """Create containers based on YAML configuration data"""
        return self.container_manager.create_containers()

    def update_config_status(self, status_text):
        """Update UI to reflect config status"""
        # Assuming you have a status label
        if hasattr(self, "config_status_label"):
            self.config_status_label.setText(status_text)

    def start_recording(self):
        """Start recording process"""
        try:
            if self.is_recording:
                logger.warning("Recording already in progress")
                return False

            if not self.yaml_parser.is_loaded():
                logger.error("No configuration loaded - cannot start recording")
                return False

            # Create subscribers now (at recording start, not at config load)
            logger.info("Creating ROS subscribers for recording...")
            subscriber_results = self.topic_subscribers.create_subscribers_for_containers(
                self.topic_containers.containers
            )
            successful = sum(subscriber_results.values())
            logger.info(
                f"Subscribers created: {successful}/{len(self.topic_containers.containers)}"
            )

            # Start recording with DataRecorder
            dataset_name = self.yaml_parser.get_dataset_name()
            demo_name = f"{dataset_name}_{int(time.time())}"
            success = self.data_recorder.start_recording(
                self.topic_subscribers, demo_name
            )

            if success:
                CustomMessageBox.info(
                    "Started Recording",
                    "Please check Record Page to visualize the topics.",
                )

                # IMPORTANT: Set RecordPage recording state
                self.is_recording = True
                self.recording_start_time = time.time()

                # Update container statuses
                self.container_manager.set_statuses_by_data(
                    "Recording (Data Available)", "Recording (Waiting for Data)"
                )

                logger.info(f"Started recording: {demo_name}")

                # Update UI buttons if they exist
                if hasattr(self.ui, "btn_start_record"):
                    self.ui.btn_start_record.setEnabled(False)
                if hasattr(self.ui, "btn_pause_record"):
                    self.ui.btn_pause_record.setEnabled(True)
                if hasattr(self.ui, "btn_stop_record"):
                    self.ui.btn_stop_record.setEnabled(True)

                return True
            else:
                logger.error("Failed to start recording")
                CustomMessageBox.error(
                    "Recording Error",
                    "Could not start recording. Please check the config and topics.",
                )
                return False

        except TfTopicSubscriptionError as e:
            # A configured custom tf_topic/tf_static_topic could not be
            # subscribed to, or never produced the requested transform.
            # Recording must NOT start on a silent fallback to the default
            # /tf, /tf_static tree — fail loudly and visibly instead.
            logger.error(f"Custom TF topic subscription failed: {e}")
            CustomMessageBox.error(
                "TF Topic Error",
                f"A configured TF topic could not be read:\n\n{e}\n\n"
                "Recording was NOT started — it will not silently fall back "
                "to the default /tf, /tf_static topics. Check that the "
                "topic is being published and that tf_topic/tf_static_topic "
                "in your config are correct.",
            )
            return False
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            return False

    def pause_recording(self):
        """Pause recording process"""
        try:
            if not self.is_recording:
                logger.warning("No recording in progress to pause")
                return False

            success = self.data_recorder.pause_recording()

            if success:
                # Update container statuses
                self.container_manager.set_status_for_all("Paused")

                logger.info("Recording paused")

                # Update UI buttons
                if hasattr(self.ui, "btn_pause_record"):
                    self.ui.btn_pause_record.setText("Resume")
                    # Disconnect old connection and connect resume
                    self.ui.btn_pause_record.clicked.disconnect()
                    self.ui.btn_pause_record.clicked.connect(self.resume_recording)

                return True
            else:
                logger.error("Failed to pause recording")
                return False

        except Exception as e:
            logger.error(f"Error pausing recording: {e}")
            return False

    def resume_recording(self):
        """Resume recording process"""
        try:
            success = self.data_recorder.resume_recording()

            if success:
                # Update container statuses
                self.container_manager.set_statuses_by_data(
                    "Recording (Data Available)", "Recording (Waiting for Data)"
                )

                logger.info("Recording resumed")

                # Update UI buttons
                if hasattr(self.ui, "btn_pause_record"):
                    self.ui.btn_pause_record.setText("Pause")
                    # Disconnect old connection and connect pause
                    self.ui.btn_pause_record.clicked.disconnect()
                    self.ui.btn_pause_record.clicked.connect(self.pause_recording)

                return True
            else:
                logger.error("Failed to resume recording")
                return False

        except Exception as e:
            logger.error(f"Error resuming recording: {e}")
            return False

    def stop_recording(self):
        """Stop recording process"""
        try:
            if not self.is_recording:
                logger.warning("No recording in progress")
                return False

            # Stop recording
            success = self.data_recorder.stop_recording()

            if success:
                self.is_recording = False
                recording_duration = (
                    time.time() - self.recording_start_time
                    if self.recording_start_time
                    else 0
                )

                # Track this demo for the session and run an automatic viability check
                file_path = self.data_recorder.h5_file_path
                if file_path:
                    self.session_h5_files.append(file_path)
                    try:
                        self.last_validation_result = self._demo_validator.validate(file_path)
                        if self.last_validation_result.overall_status == CheckStatus.FAIL:
                            CustomMessageBox.warning(
                                f"Demo saved but failed validation:\n\n"
                                f"{self.last_validation_result.summary_text()}",
                                title="Demo Validation Failed",
                            )
                    except Exception as e:
                        logger.error(f"Error validating demo: {e}")

                # Destroy all subscribers — they will be re-created on next recording start
                self.topic_subscribers.clear_all_subscribers()
                logger.info("Subscribers destroyed after recording stopped")

                # Update container statuses
                self.container_manager.set_statuses_by_data(
                    "Stopped (Data Available)", "Stopped (No Data)"
                )

                logger.info(
                    f"Stopped recording. Duration: {recording_duration:.2f} seconds"
                )

                # Update UI buttons
                if hasattr(self.ui, "btn_start_record"):
                    self.ui.btn_start_record.setEnabled(True)
                if hasattr(self.ui, "btn_pause_record"):
                    self.ui.btn_pause_record.setEnabled(False)
                    self.ui.btn_pause_record.setText("Pause")  # Reset text
                    # Reconnect to pause function
                    self.ui.btn_pause_record.clicked.disconnect()
                    self.ui.btn_pause_record.clicked.connect(self.pause_recording)
                if hasattr(self.ui, "btn_stop_record"):
                    self.ui.btn_stop_record.setEnabled(False)

                return True
            else:
                logger.error("Failed to stop recording")
                return False

        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            return False

    def remove_session_file(self, path: str):
        """Prune a demo file from this session's tracked recordings (e.g. after deletion)."""
        if path in self.session_h5_files:
            self.session_h5_files.remove(path)

    def get_container_data(self, input_name):
        """Get the latest ROS data for a container by input name"""
        return self.container_manager.get_container_data(input_name)

    def get_container_by_input_name(self, input_name):
        """Get container by input name"""
        return self.container_manager.get_container_by_input_name(input_name)

    def get_all_topic_data(self):
        """Get all current topic data"""
        return self.topic_subscribers.get_content()

    def get_recording_summary(self):
        """Get a comprehensive summary of the recording setup"""
        if not self.yaml_parser.is_loaded():
            return None

        # Get subscriber summary
        subscriber_summary = self.topic_subscribers.get_all_data_summary()

        # Get recording status
        recording_status = self.data_recorder.get_recording_status()

        config = {
            "input_data": self.yaml_parser.get_input_data(),
            "output_data": self.yaml_parser.get_output_data(),
            "config_data": self.yaml_parser.get_config_data(),
            "topic_data": self.yaml_parser.get_topic_data(),
            "prefix_path": self.yaml_parser.get_prefix_path(),
            "containers": self.get_all_container_info(),
            "subscribers": subscriber_summary,
            "recording_status": recording_status,
            "topics_with_data": [
                cid
                for cid in subscriber_summary["subscriber_details"]
                if cid.get("has_data", False)
            ],
        }

        return config

    def get_all_container_info(self):
        """Get information about all containers"""
        return self.container_manager.get_all_container_info()

    def wait_for_all_data(self, timeout=10.0):
        """Wait for all containers to receive data"""
        return self.container_manager.wait_for_all_data(timeout)

    def restart_failed_subscribers(self):
        """Restart subscribers that failed to create or stopped working"""
        return self.container_manager.restart_failed_subscribers()

    def update_container_statuses(self):
        """Update all container statuses based on current subscriber state"""
        self.container_manager.update_container_statuses()

    def force_ui_refresh(self):
        """Force UI refresh"""
        self.container_manager.force_ui_refresh()

    def on_window_resize(self):
        """Handle window resize events"""
        self.container_manager.on_window_resize()

    def set_visualization_rate(self, rate_hz):
        """Set the visualization update rate"""
        self.container_visualizer.set_update_rate(rate_hz)

    def setup_connections(self):
        """Setup all button connections"""
        try:
            # Container management
            if hasattr(self.ui, "btn_refresh_containers"):
                self.ui.btn_refresh_containers.clicked.connect(self._create_containers)

            if hasattr(self.ui, "btn_restart_subscribers"):
                self.ui.btn_restart_subscribers.clicked.connect(
                    self.restart_failed_subscribers
                )

            if hasattr(self.ui, "btn_update_status"):
                self.ui.btn_update_status.clicked.connect(
                    self.update_container_statuses
                )

            ## Buttons to start, record and pause
            if hasattr(self.ui, "btn_start_record"):
                self.ui.btn_start_record.clicked.connect(self.start_recording)
                self.ui.btn_start_record.setEnabled(False)

            # Connect pause button to pause_recording method
            if hasattr(self.ui, "btn_pause_record"):
                self.ui.btn_pause_record.clicked.connect(self.pause_recording)
                self.ui.btn_pause_record.setEnabled(False)
                self.ui.btn_pause_record.setText("Pause")  # Set initial text

            if hasattr(self.ui, "btn_stop_record"):
                self.ui.btn_stop_record.clicked.connect(self.stop_recording)
                self.ui.btn_stop_record.setEnabled(False)

        except Exception as e:
            logger.error("Error setting up connections.")

    def debug_containers(self):
        """Debug method to inspect container widgets"""
        self.container_manager.debug_containers()
