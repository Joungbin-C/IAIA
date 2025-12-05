# coding=utf-8
import sys
import cv2
import numpy as np
from ultralytics import YOLO
import PySpin
import socket  # 통신 주석 처리

# PyQt5 Imports
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QVBoxLayout, QHBoxLayout, QGridLayout, QTextEdit,
                             QGroupBox, QScrollArea, QPushButton)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import (QThread, pyqtSignal, pyqtSlot, Qt,
                          QMutex, QWaitCondition)

HOST = '192.168.0.18'
PORT = 9999

EDGE_RING_THICKNESS = 10
CANNY_LOW, CANNY_HIGH = 80, 160
EDGE_RATIO_THRESHOLD = 3.0
NUM_SECTORS = 24
SECTOR_CRACK_THRESHOLD = 0.4
OUTER_MIN_RADIUS = 280
OUTER_MAX_RADIUS = 350
INNER_MIN_RADIUS = 180
INNER_MAX_RADIUS = 250
CENTER_OFFSET_TOLERANCE = 50
MIN_PIXEL_AREA_THRESHOLD = 100000


def create_connection(host=HOST, port=PORT, timeout=10):
    try:
        s = socket.create_connection((host, port), timeout=timeout)
        print("통신 연결 성공")
        return s
    except Exception as e:
        print(f"통신 연결 실패: {e}")
        return None


def send_flag(sock, flag):
    try:
        sock.sendall(flag.encode('utf-8'))
        print(f"Flag '{flag}' 전송 완료")
    except Exception as e:
        print(f"전송 오류: {e}")


class ProcessingThread(QThread):
    change_pixmap_cell = pyqtSignal(np.ndarray)
    change_pixmap_surface = pyqtSignal(np.ndarray)
    change_pixmap_edge = pyqtSignal(np.ndarray)
    log_message = pyqtSignal(str)
    update_status_signal = pyqtSignal(int, str, str, str)

    def __init__(self):
        super().__init__()
        self._running = True
        self._paused = False
        self.mutex = QMutex()
        self.pause_cond = QWaitCondition()

    def stop(self):
        self.mutex.lock()
        self._running = False
        self.mutex.unlock()
        self.pause_cond.wakeAll()

    def pause(self):
        self.mutex.lock()
        self._paused = True
        self.mutex.unlock()

    def resume(self):
        self.mutex.lock()
        self._paused = False
        self.mutex.unlock()
        self.pause_cond.wakeAll()


    def run(self):
        system = None
        cam_list = None
        cam = None
        cap = None
        # sock = None # 통신 주석 처리

        try:
            system = PySpin.System.GetInstance()
            version = system.GetLibraryVersion()
            self.log_message.emit(
                'Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

            cam_list = system.GetCameras()
            if cam_list.GetSize() == 0:
                self.log_message.emit('No cameras found')
                cam_list.Clear();
                system.ReleaseInstance();
                return

            cam = cam_list[0]
            self.log_message.emit("Spinnaker 카메라 연결 완료")
            cam.Init()
            nodemap = cam.GetNodeMap()

            # sock = create_connection()
            # if not sock:
            #     self.log_message.emit("통신 연결 실패. 스레드 종료.")
            #     return

            sNodemap = cam.GetTLStreamNodeMap()
            node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
            if PySpin.IsReadable(node_bufferhandling_mode) and PySpin.IsWritable(node_bufferhandling_mode):
                node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
                if PySpin.IsReadable(node_newestonly):
                    node_bufferhandling_mode.SetIntValue(node_newestonly.GetValue())

            node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
            if PySpin.IsReadable(node_acquisition_mode) and PySpin.IsWritable(node_acquisition_mode):
                node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
                if PySpin.IsReadable(node_acquisition_mode_continuous):
                    node_acquisition_mode.SetIntValue(node_acquisition_mode_continuous.GetValue())
                    self.log_message.emit('Acquisition mode set to continuous...')

            surface_model_path = 'C:/Users/joung/Downloads/spinnaker_python-4.2.0.88-cp310-cp310-win_amd64/train/runs/train5/weights/best.pt'
            try:
                surface_model = YOLO(surface_model_path)
                self.log_message.emit(f"Surface model loaded: {surface_model_path}")
            except Exception as e:
                self.log_message.emit(f"Error loading YOLO: {e}");
                return

            processor = PySpin.ImageProcessor()
            cam.BeginAcquisition()

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.log_message.emit("Cannot Open USB Cam")
                cam.EndAcquisition();
                return

            is_object_centered_previously = False
            current_cell_index = 1
            self.update_status_signal.emit(1, "WAITING", "---", "")
            self.update_status_signal.emit(2, "---", "---", "")

            while self._running:
                self.mutex.lock()
                while self._paused and self._running:
                    self.pause_cond.wait(self.mutex)
                self.mutex.unlock()

                if not self._running:
                    break

                try:
                    image_result = cam.GetNextImage(1000)
                    if image_result.IsIncomplete():
                        image_result.Release();
                        continue

                    image_converted = processor.Convert(image_result, PySpin.PixelFormat_BGR8)
                    color_image_data = image_converted.GetNDArray().copy()
                    h, w, _ = color_image_data.shape
                    roi = color_image_data[h // 2 - 500: h // 2 + 500, w // 2 - 500: w // 2 + 500]
                    gray_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                    h_roi, w_roi = gray_image.shape
                    roi_center_x = w_roi // 2
                    blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 2)
                    annotated_frame = roi.copy()

                    msg = "OK"
                    failure_reason = ""
                    should_run_surface_detection = False

                    outer_circles = cv2.HoughCircles(
                        image=blurred_image, method=cv2.HOUGH_GRADIENT, dp=1,
                        minDist=gray_image.shape[0] // 4, param1=100, param2=30,
                        minRadius=OUTER_MIN_RADIUS, maxRadius=OUTER_MAX_RADIUS
                    )

                    if outer_circles is not None:
                        circles = np.uint16(np.around(outer_circles))
                        largest_circle = max(circles[0, :], key=lambda c: c[2])
                        cx, cy, r_outer = largest_circle
                        center = (cx, cy)
                        circle_area = np.pi * (r_outer ** 2)

                        if circle_area >= MIN_PIXEL_AREA_THRESHOLD:
                            should_run_surface_detection = True

                            # --- 균열 검출 로직 ---
                            mask_outer = np.zeros_like(gray_image)
                            cv2.circle(mask_outer, center, r_outer + EDGE_RING_THICKNESS, 255, -1)
                            cv2.circle(mask_outer, center, r_outer - EDGE_RING_THICKNESS, 0, -1)
                            ring_roi = cv2.bitwise_and(gray_image, mask_outer)
                            edges = cv2.Canny(ring_roi, CANNY_LOW, CANNY_HIGH)
                            self.change_pixmap_edge.emit(edges)

                            edge_count = np.count_nonzero(edges)
                            circle_circumference = 2 * np.pi * r_outer
                            edge_ratio = edge_count / circle_circumference

                            if edge_ratio < EDGE_RATIO_THRESHOLD:
                                result_text = "NG - Crack detected"
                                color = (0, 0, 255)
                                msg = "NOK";
                                failure_reason = "Crack detected"
                                angle_step = 360 / NUM_SECTORS
                                ideal_arc_length_sector = circle_circumference / NUM_SECTORS
                                for i in range(NUM_SECTORS):
                                    start_angle = int(i * angle_step);
                                    end_angle = int((i + 1) * angle_step)
                                    ring_sector_mask = np.zeros_like(gray_image)
                                    cv2.ellipse(ring_sector_mask, center,
                                                (r_outer + EDGE_RING_THICKNESS, r_outer + EDGE_RING_THICKNESS), 0,
                                                start_angle, end_angle, 255, -1)
                                    cv2.ellipse(ring_sector_mask, center,
                                                (r_outer - EDGE_RING_THICKNESS, r_outer - EDGE_RING_THICKNESS), 0,
                                                start_angle, end_angle, 0, -1)
                                    sector_edges = cv2.bitwise_and(edges, ring_sector_mask)
                                    edge_count_in_sector = np.count_nonzero(sector_edges)
                                    sector_ratio = edge_count_in_sector / ideal_arc_length_sector
                                    if sector_ratio < SECTOR_CRACK_THRESHOLD:
                                        cv2.ellipse(annotated_frame, center, (r_outer, r_outer), 0, start_angle,
                                                    end_angle, color, 5)
                            else:
                                result_text = "OK - No crack";
                                color = (0, 255, 0)

                            cv2.circle(annotated_frame, center, r_outer, color, 3);
                            cv2.circle(annotated_frame, center, 2, (255, 255, 255), 3)

                            # --- 두께 측정 로직 ---
                            inner_circles = cv2.HoughCircles(
                                image=blurred_image, method=cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                param1=100, param2=30, minRadius=INNER_MIN_RADIUS, maxRadius=INNER_MAX_RADIUS
                            )
                            if inner_circles is not None:
                                circles_inner = np.uint16(np.around(inner_circles))
                                largest_inner_circle = max(circles_inner[0, :], key=lambda c: c[2])
                                cx_inner, cy_inner, r_inner = largest_inner_circle
                                center_inner = (cx_inner, cy_inner)
                                cv2.circle(annotated_frame, center_inner, r_inner, (0, 255, 255), 3)
                                thickness = r_outer - r_inner
                                thickness_text = f"Thickness: {thickness} px"
                            else:
                                continue
                        else:
                            msg = "NOK";
                            failure_reason = "Circle too small"
                            should_run_surface_detection = False
                    else:
                        msg = "NOK";
                        failure_reason = "No circle found"
                        should_run_surface_detection = False

                    self.change_pixmap_cell.emit(annotated_frame)

                    # --- USB 카메라 처리 ---
                    ret, frame = cap.read()
                    if not ret:
                        self.log_message.emit("USB Cam read error");
                        break
                    h_usb, w_usb, _ = frame.shape
                    frame_roi = frame[h_usb // 2 - 100:h_usb // 2 + 150, w_usb // 2 - 150:w_usb // 2 + 150]

                    surface_display_frame = frame_roi.copy()

                    if should_run_surface_detection:
                        surface_results = surface_model(frame_roi, verbose=False)
                        surface_display_frame = surface_results[0].plot()
                        num_faults = len(surface_results[0])

                        if num_faults >= 3:
                            if msg == "OK":
                                failure_reason = f"Surface faults >= {num_faults}"
                            else:
                                failure_reason += f" & Surface faults >= {num_faults}"
                            msg = "NOK"

                        is_centered_x = abs(int(cx) - roi_center_x) < CENTER_OFFSET_TOLERANCE

                        if is_centered_x and not is_object_centered_previously:
                            is_object_centered_previously = True
                            numeric_msg = ""

                            result_msg = msg
                            reason_msg = failure_reason

                            if current_cell_index == 1:
                                numeric_msg = "1" if msg == "OK" else "2"
                                log_str = f"--- Cell 1 DETECTED --- Result: {msg}."
                                if msg == "NOK": log_str += f" Reason: {failure_reason}."
                                log_str += f" Sending Flag: {numeric_msg}"
                                self.log_message.emit(log_str)
                                # send_flag(sock, numeric_msg) # 통신 주석 처리

                                self.update_status_signal.emit(1, "DETECTED", result_msg, reason_msg)
                                self.update_status_signal.emit(2, "WAITING", "__NO_CHANGE__", "__NO_CHANGE__")
                                current_cell_index = 2

                            elif current_cell_index == 2:
                                numeric_msg = "3" if msg == "OK" else "4"
                                log_str = f"--- Cell 2 DETECTED --- Result: {msg}."
                                if msg == "NOK": log_str += f" Reason: {failure_reason}."
                                log_str += f" Sending Flag: {numeric_msg}"
                                self.log_message.emit(log_str)
                                # send_flag(sock, numeric_msg) # 통신 주석 처리

                                self.update_status_signal.emit(2, "DETECTED", result_msg, reason_msg)
                                self.update_status_signal.emit(1, "WAITING", "__NO_CHANGE__", "__NO_CHANGE__")
                                current_cell_index = 1

                        elif not is_centered_x:
                            if is_object_centered_previously:
                                self.log_message.emit(
                                    f"Object has moved out. Waiting for Cell {current_cell_index}.")
                            is_object_centered_previously = False

                    else:
                        if is_object_centered_previously:
                            self.log_message.emit("Object detection failed (no circle). Resetting center flag.")
                            is_object_centered_previously = False

                    self.change_pixmap_surface.emit(surface_display_frame)
                    image_result.Release()
                    # ... (이미지 처리 로직 끝) ...

                except PySpin.SpinnakerException as ex:
                    self.log_message.emit(f"Spinnaker exception in loop: {ex}");
                    continue
                except Exception as e:
                    self.log_message.emit(f"General exception in loop: {e}");
                    continue

        except PySpin.SpinnakerException as ex:
            self.log_message.emit(f'Error during setup: {ex}')
        except Exception as e:
            self.log_message.emit(f'Error during setup: {e}')
        finally:
            self.log_message.emit("Cleaning up resources...")
            if cam and cam.IsStreaming(): cam.EndAcquisition()
            if cap and cap.isOpened(): cap.release()
            if cam and cam.IsInitialized(): cam.DeInit()
            # if sock: sock.close() # 통신 주석 처리
            if cam_list: cam_list.Clear()
            if system: system.ReleaseInstance()
            self.log_message.emit("Cleanup complete. Thread finished.")


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Vision Inspection")
        self.setGeometry(100, 100, 1600, 900)

        self.is_paused = False

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # --- 왼쪽: 이미지 그리드 레이아웃 ---
        image_grid_layout = QGridLayout()

        self.label_cell = self.create_display_label()
        image_grid_layout.addWidget(self.create_title_label("Cell Detection (Circle)"), 0, 0)
        image_grid_layout.addWidget(self.label_cell, 1, 0)

        self.label_surface = self.create_display_label()
        image_grid_layout.addWidget(self.create_title_label("Surface Detecting"), 0, 1)
        image_grid_layout.addWidget(self.label_surface, 1, 1)

        self.label_edge = self.create_display_label()
        image_grid_layout.addWidget(self.create_title_label("Edge Ring (Crack Check)"), 2, 0)
        image_grid_layout.addWidget(self.label_edge, 3, 0)

        cell_status_layout = QVBoxLayout()
        cell_status_layout.addWidget(self.create_status_group("Cell 1"))
        cell_status_layout.addWidget(self.create_status_group("Cell 2"))
        cell_status_layout.addStretch()

        image_grid_layout.addLayout(cell_status_layout, 2, 1, 2, 1)

        main_layout.addLayout(image_grid_layout, 3)

        # --- 오른쪽: 로그 및 제어 버튼 ---
        right_pane_layout = QVBoxLayout()

        right_pane_layout.addWidget(self.create_title_label("Logs"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(200)
        right_pane_layout.addWidget(self.log_box)  # 로그 박스

        # --- 시작/일시정지/정지 버튼 ---
        self.start_button = QPushButton("Start")
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white; height: 30px; font-weight: bold;")

        self.pause_button = QPushButton("Pause")
        self.pause_button.setStyleSheet(
            "background-color: #FFC107; color: black; height: 30px; font-weight: bold;")

        self.stop_button = QPushButton("Stop")
        self.stop_button.setStyleSheet("background-color: #F44336; color: white; height: 30px; font-weight: bold;")

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.stop_button)

        right_pane_layout.addLayout(button_layout)

        main_layout.addLayout(right_pane_layout, 1)

        # --- 스레드 생성 ---
        self.thread = ProcessingThread()
        self.connect_signals()

        # --- 버튼 연결 및 초기 상태 설정 ---
        self.start_button.clicked.connect(self.start_processing)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.stop_button.clicked.connect(self.stop_processing)

        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)

    def create_status_group(self, title):
        group_box = QGroupBox(title)
        group_box.setStyleSheet("QGroupBox { font-size: 16px; font-weight: bold; margin-top: 10px; }")
        v_layout = QVBoxLayout()

        # --- State ---
        state_label_title = QLabel("State:")
        state_label_title.setStyleSheet("font-size: 14px; font-weight: normal;")
        state_label_value = QLabel("---")

        # --- Result ---
        result_label_title = QLabel("Result:")
        result_label_title.setStyleSheet("font-size: 14px; font-weight: normal; margin-top: 5px;")
        result_label_value = QLabel("---")

        # --- Reason ---
        reason_label_title = QLabel("Reason:")
        reason_label_title.setStyleSheet("font-size: 14px; font-weight: normal; margin-top: 5px;")
        reason_label_value = QLabel("---")

        v_layout.addWidget(state_label_title)
        v_layout.addWidget(state_label_value)
        v_layout.addWidget(result_label_title)
        v_layout.addWidget(result_label_value)
        v_layout.addWidget(reason_label_title)
        v_layout.addWidget(reason_label_value)
        v_layout.addStretch()

        group_box.setLayout(v_layout)

        if title == "Cell 1":
            self.label_cell1_status = state_label_value
            self.label_cell1_result = result_label_value
            self.label_cell1_reason_title = reason_label_title
            self.label_cell1_reason_value = reason_label_value
        else:
            self.label_cell2_status = state_label_value
            self.label_cell2_result = result_label_value
            self.label_cell2_reason_title = reason_label_title
            self.label_cell2_reason_value = reason_label_value

        reason_label_title.hide()
        reason_label_value.hide()

        return group_box

    def create_display_label(self):
        label = QLabel()
        label.setMinimumSize(480, 360)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("border: 1px solid #555; background-color: #333;")
        return label

    def create_title_label(self, text):
        label = QLabel(text)
        label.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 10px;")
        label.setAlignment(Qt.AlignCenter)
        return label

    def connect_signals(self):
        self.thread.change_pixmap_cell.connect(lambda img: self.update_image(img, self.label_cell))
        self.thread.change_pixmap_surface.connect(lambda img: self.update_image(img, self.label_surface))
        self.thread.change_pixmap_edge.connect(lambda img: self.update_image(img, self.label_edge, is_gray=True))
        self.thread.log_message.connect(self.update_log)
        self.thread.update_status_signal.connect(self.update_status_display)

        self.thread.finished.connect(self.on_thread_finished)

    def closeEvent(self, event):
        self.update_log("Main window closing. Stopping thread...")
        if self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()
        event.accept()

    @pyqtSlot(str)
    def update_log(self, message):
        self.log_box.append(message)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    @pyqtSlot(int, str, str, str)
    def update_status_display(self, cell_num, status, result, reason):
        status_label = None
        result_label = None
        reason_title_label = None
        reason_value_label = None

        if cell_num == 1:
            status_label = self.label_cell1_status
            result_label = self.label_cell1_result
            reason_title_label = self.label_cell1_reason_title
            reason_value_label = self.label_cell1_reason_value
        elif cell_num == 2:
            status_label = self.label_cell2_status
            result_label = self.label_cell2_result
            reason_title_label = self.label_cell2_reason_title
            reason_value_label = self.label_cell2_reason_value
        else:
            return

        base_style = "font-size: 16px; padding: 5px; border-radius: 5px;"

        if status == "DETECTED":
            status_color_style = "background-color: #4CAF50; color: white;"
        elif status == "WAITING":
            status_color_style = "background-color: #FFC107; color: black;"
        else:
            status_color_style = "background-color: #607D8B; color: white;"
        status_label.setStyleSheet(base_style + status_color_style)
        status_label.setText(status)

        if result != "__NO_CHANGE__":
            # --- 2a. Result 업데이트 ---
            if "NOK" in result:
                result_color_style = "background-color: #F44336; color: white;"
            elif "OK" in result:
                result_color_style = "background-color: #4CAF50; color: white;"
            else:  # "---"
                result_color_style = "background-color: #607D8B; color: white;"
            result_label.setStyleSheet(base_style + result_color_style)
            result_label.setText(result)

            if result == "NOK":
                reason_value_label.setText(reason)
                reason_value_label.setStyleSheet(
                    base_style + "background-color: #333; color: #F44336; font-weight: bold;")
                reason_title_label.show()
                reason_value_label.show()
            else:
                reason_value_label.setText("---")
                reason_title_label.hide()
                reason_value_label.hide()

    @pyqtSlot(np.ndarray, QLabel, bool)
    def update_image(self, cv_img, label, is_gray=False):
        try:
            if is_gray:
                h, w = cv_img.shape
                bytes_per_line = w
                qt_img_format = QImage.Format_Grayscale8
                qt_img = QImage(cv_img.data, w, h, bytes_per_line, qt_img_format)
            else:
                rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_img_format = QImage.Format_RGB888
                qt_img = QImage(rgb_image.data, w, h, bytes_per_line, qt_img_format)

            pixmap = QPixmap.fromImage(qt_img)
            scaled_pixmap = pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Error updating image: {e}")


    @pyqtSlot()
    def start_processing(self):
        if self.thread.isFinished():
            self.update_log("Creating new processing thread...")
            self.thread = ProcessingThread()
            self.connect_signals()

        if not self.thread.isRunning():
            self.update_log("Starting processing thread...")
            self.thread.start()
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.pause_button.setEnabled(True)
            self.is_paused = False
            self.pause_button.setText("Pause")
        else:
            self.update_log("Thread is already running.")

    @pyqtSlot()
    def toggle_pause(self):
        """ '일시 정지' / '다시 시작' 버튼 클릭 """
        if not self.thread.isRunning():
            return

        self.is_paused = not self.is_paused

        if self.is_paused:
            self.thread.pause()
            self.update_log("Processing paused.")
            self.pause_button.setText("Resume")
        else:
            self.thread.resume()
            self.update_log("Processing resumed.")
            self.pause_button.setText("Pause")

    @pyqtSlot()
    def stop_processing(self):
        if self.thread.isRunning():
            self.update_log("Signaling thread to stop...")
            self.thread.stop()
            self.stop_button.setEnabled(False)
            self.pause_button.setEnabled(False)
        else:
            self.update_log("Thread is not running.")

    @pyqtSlot()
    def on_thread_finished(self):
        self.update_log("Processing thread has finished.")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.is_paused = False
        self.pause_button.setText("Pause")


if __name__ == '__main__':
    app = QApplication(sys.argv)

    app.setStyle('Fusion')
    try:
        from PyQt5.QtGui import QPalette
        from PyQt5.QtGui import QColor

        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        app.setPalette(palette)
    except ImportError:
        pass

    ex = App()
    ex.show()
    sys.exit(app.exec_())