import cv2
import os
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QGraphicsView, QGraphicsScene
from PyQt5.QtGui import QImage, QPixmap, QPainter, QWheelEvent, QMouseEvent
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtWidgets import QPushButton, QRadioButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QSpinBox, QInputDialog

class CustomGraphicsView(QGraphicsView):
    def __init__(self, editor):
        super(CustomGraphicsView, self).__init__()

        self.editor = editor
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setRenderHint(QPainter.TextAntialiasing)

        self.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, True)
        self.setOptimizationFlag(QGraphicsView.DontSavePainterState, True)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setInteractive(True)
        # 启用鼠标移动追踪（无需按键按下）
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.image_item = None

    def set_image(self, q_img):
        pixmap = QPixmap.fromImage(q_img)
        if self.image_item:
            self.image_item.setPixmap(pixmap)
        else:
            self.image_item = self.scene.addPixmap(pixmap)
            self.setSceneRect(QRectF(pixmap.rect()))

    def wheelEvent(self, event: QWheelEvent):
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        old_pos = self.mapToScene(event.pos())
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor
        self.scale(zoom_factor, zoom_factor)
        new_pos = self.mapToScene(event.pos())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())
    
    def imshow(self, img):
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.set_image(q_img)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        pos = event.pos()
        pos_in_item = self.mapToScene(pos) - self.image_item.pos()
        x, y = pos_in_item.x(), pos_in_item.y()
        if event.button() == Qt.LeftButton:
            label = 1
        elif event.button() == Qt.RightButton:
            label = 0        
        self.editor.add_click([int(x), int(y)], label)
        self.imshow(self.editor.display)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.image_item is None:
            return
        pos = event.pos()
        pos_in_item = self.mapToScene(pos) - self.image_item.pos()
        x, y = int(pos_in_item.x()), int(pos_in_item.y())
        # 仅当坐标在图像范围内时处理
        if x < 0 or y < 0:
            return
        # 更新悬停并按需刷新
        changed = self.editor.update_hover(x, y)
        if changed:
            self.imshow(self.editor.display)
    
class ApplicationInterface(QWidget):
    def __init__(self, app, editor, panel_size=(1920, 1080)):
        super(ApplicationInterface, self).__init__()

        self.app = app
        self.editor = editor
        self.panel_size = panel_size

        self.layout = QVBoxLayout()

        self.top_bar = self.get_top_bar()
        self.layout.addWidget(self.top_bar)

        
        self.main_window = QHBoxLayout()
        
        self.graphics_view = CustomGraphicsView(self.editor)
        self.main_window.addWidget(self.graphics_view)

        self.panel = self.get_side_panel()
        self.main_window.addWidget(self.panel)
        self.layout.addLayout(self.main_window)

        self.setLayout(self.layout)

        self.graphics_view.imshow(self.editor.display)
        # 初始化顶部信息
        self._update_info_label()

    def _update_info_label(self):
        total = self.editor.get_image_count()
        if total == 0:
            if hasattr(self, "info_label"):
                self.info_label.setText("无图片")
            return
        idx = self.editor.get_current_index()
        name = self.editor.get_current_filename()
        if hasattr(self, "info_label"):
            self.info_label.setText(f"{idx+1}/{total}: {name}")

    def _refresh_view(self):
        self.graphics_view.imshow(self.editor.display)
        self._update_info_label()
    
    def reset(self):
        self.editor.reset()
        self._refresh_view()    

    def add(self):
        self.editor.save_ann()
        self.editor.reset()
        self._refresh_view()    

    def delet(self):
        self.editor.delet_ann()
        self.editor.reset()
        self._refresh_view()   
        
    def next_image(self):
        self.editor.next_image()
        self._refresh_view()
        self.editor.save()

    def prev_image(self):
        self.editor.prev_image()
        self._refresh_view()    

    def toggle(self):
        self.editor.toggle()
        self._refresh_view()    

    def transparency_up(self):
        self.editor.step_up_transparency()
        self._refresh_view()

    def transparency_down(self):
        self.editor.step_down_transparency()
        self._refresh_view()
    
    def save_all(self):
        self.editor.save()

    def get_top_bar(self):
        top_bar = QWidget()
        button_layout = QHBoxLayout(top_bar)
        self.layout.addLayout(button_layout)
        # 放大菜单栏字体
        font = top_bar.font()
        font.setPointSize(14)  # 你可以改为更大/更小，如 12/16
        top_bar.setFont(font)
        buttons = [
            ("添加对象", lambda: self.add()),
            ("撤销对象", lambda: self.delet()),
            ("重置", lambda: self.reset()),
            ("前一张", lambda: self.prev_image()),
            ("下一张", lambda: self.next_image()),
            ("显示已标注信息", lambda: self.toggle()),
            ("悬停显示标签: 关", lambda: self.toggle_hover_mode()),
            ("调高透明度", lambda: self.transparency_up()),
            ("调低透明度", lambda: self.transparency_down()),
            ("保存", lambda: self.save_all()), 
        ]
        for button, lmb in buttons:
            bt = QPushButton(button)
            bt.clicked.connect(lmb)
            bt.setMinimumHeight(34)
            button_layout.addWidget(bt)

        # 跳页控件与当前信息
        button_layout.addStretch(1)

        # 信息标签
        self.info_label = QLabel("")
        button_layout.addWidget(self.info_label)

        # 跳转到序号（1-based）
        jump_label = QLabel(" 跳到序号: ")
        button_layout.addWidget(jump_label)

        self.jump_spin = QSpinBox()
        total = self.editor.get_image_count()
        self.jump_spin.setMinimum(1)
        self.jump_spin.setMaximum(max(1, total))
        self.jump_spin.setValue(self.editor.get_current_index() + 1)
        button_layout.addWidget(self.jump_spin)

        self.jump_btn = QPushButton("跳转")
        self.jump_btn.clicked.connect(self.jump_to_index)
        self.jump_spin.editingFinished.connect(self.jump_to_index)
        button_layout.addWidget(self.jump_btn)

        return top_bar

    def get_side_panel(self):
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        # 放大侧栏字体
        font = panel.font()
        font.setPointSize(14)
        panel.setFont(font)
        categories = self.editor.get_categories()
        for category in categories:
            # label = QPushButton(category)
            # label.clicked.connect(lambda: self.editor.select_category(category))
            # panel_layout.addWidget(label)

            label = QRadioButton(category)
            label.setMinimumHeight(28)
            # sender传入点击的字符
            label.toggled.connect(lambda: self.editor.select_category(self.sender().text()))
            panel_layout.addWidget(label)
        return panel

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.app.quit()
        if event.key() == Qt.Key_A:
            self.prev_image()
        if event.key() == Qt.Key_D:
            self.next_image()
        if event.key() == Qt.Key_PageUp:
            self.prev_image()
        if event.key() == Qt.Key_PageDown:
            self.next_image()
        if event.key() == Qt.Key_K:
            self.transparency_down()
        if event.key() == Qt.Key_L:
            self.transparency_up()
        if event.key() == Qt.Key_N:
            self.add()
        if event.key() == Qt.Key_R:
            self.reset()
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_S:
            self.save_all()
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_Z:
            self.delet()
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_G:
            total = self.editor.get_image_count()
            if total > 0:
                val, ok = QInputDialog.getInt(self, "跳转", f"输入序号 (1 - {total})", value=self.editor.get_current_index()+1, min=1, max=total)
                if ok:
                    self.jump_spin.setValue(val)
                    self.jump_to_index()
        # elif event.key() == Qt.Key_Space:
        #     # Do something if the space bar is pressed
        #     pass

    def toggle_hover_mode(self):
        # 切换 Editor 的悬停模式
        prev = self.editor.hover_mode_enabled
        self.editor.toggle_hover_mode()
        # 更新按钮文字（找到该按钮并修改文本）
        sender = self.sender()
        if isinstance(sender, QPushButton):
            now = self.editor.hover_mode_enabled
            sender.setText("悬停显示标签: 开" if now else "悬停显示标签: 关")
        # 刷新视图
        self._refresh_view()

    def jump_to_index(self):
        total = self.editor.get_image_count()
        if total == 0:
            return
        idx_1 = self.jump_spin.value()
        self.editor.go_to_image(idx_1 - 1)
        # 同步范围与视图
        self.jump_spin.blockSignals(True)
        self.jump_spin.setMaximum(max(1, self.editor.get_image_count()))
        self.jump_spin.setValue(self.editor.get_current_index() + 1)
        self.jump_spin.blockSignals(False)
        self._refresh_view()
