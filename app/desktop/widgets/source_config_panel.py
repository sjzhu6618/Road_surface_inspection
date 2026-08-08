from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
)

from app.core.schemas import InferenceConfig


class SourceConfigPanel(QFrame):
    def __init__(self, config_service=None):
        super().__init__()
        self.config_service = config_service
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        title = QLabel('任务配置')
        title.setObjectName('sectionTitle')
        subtitle = QLabel('统一管理输入源、模型、深度模型和推理参数，减少重复配置。')
        subtitle.setWordWrap(True)
        subtitle.setObjectName('sectionSubtitle')
        root.addWidget(title)
        root.addWidget(subtitle)

        self.mode_box = QComboBox()
        self.mode_box.addItems(['图片', '视频', '文件夹', '摄像头'])
        self.mode_box.currentTextChanged.connect(self._apply_mode_defaults)

        self.model_row, self.model_edit = self._path_row(
            '浏览',
            lambda: self._choose_file(self.model_edit, '选择模型权重', '权重文件 (*.pt *.onnx)'),
        )
        self.source_row, self.source_edit = self._path_row('浏览', self._choose_source)
        self.depth_row, self.depth_edit = self._path_row(
            '浏览',
            lambda: self._choose_dir(self.depth_edit, '选择深度模型目录'),
        )
        self.output_row, self.output_edit = self._path_row(
            '浏览',
            lambda: self._choose_dir(self.output_edit, '选择输出目录'),
        )
        self.font_row, self.font_edit = self._path_row(
            '浏览',
            lambda: self._choose_file(self.font_edit, '选择字体文件', '字体文件 (*.ttf *.ttc *.otf)'),
        )

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(['小型配置（快速）', '中型配置（平衡）', '大型配置（精细）'])
        self.preset_combo.setCurrentIndex(1)
        self.preset_combo.currentTextChanged.connect(self._apply_preset)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setDecimals(2)
        self.conf_spin.setRange(0.01, 0.99)
        self.conf_spin.setSingleStep(0.01)

        self.valid_conf_spin = QDoubleSpinBox()
        self.valid_conf_spin.setDecimals(2)
        self.valid_conf_spin.setRange(0.01, 0.99)
        self.valid_conf_spin.setSingleStep(0.01)

        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setDecimals(2)
        self.iou_spin.setRange(0.10, 0.95)
        self.iou_spin.setSingleStep(0.01)
        self.iou_spin.setValue(0.45)

        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(320, 1920)
        self.imgsz_spin.setSingleStep(32)

        self.frame_stride_spin = QSpinBox()
        self.frame_stride_spin.setRange(1, 30)

        self.max_frames_spin = QSpinBox()
        self.max_frames_spin.setRange(0, 999999)
        self.max_frames_spin.setSpecialValueText('不限')

        self.depth_preview_check = QCheckBox('显示深度预览')
        self.depth_preview_check.setChecked(True)
        self.road_roi_check = QCheckBox('绘制路面 ROI')
        self.road_roi_check.setChecked(True)
        self.clahe_check = QCheckBox('启用 CLAHE')
        self.clahe_check.setChecked(True)
        self.denoise_check = QCheckBox('启用去噪')
        self.denoise_check.setChecked(False)
        self.augment_check = QCheckBox('增强推理召回')
        self.augment_check.setChecked(False)
        self.save_result_check = QCheckBox('保存检测结果')
        self.save_result_check.setChecked(True)
        self.tiling_check = QCheckBox('切片推理')
        self.tiling_check.setChecked(True)

        self.device_combo = QComboBox()
        self._refresh_device_options()

        for widget in [
            self._build_source_group(),
            self._build_model_group(),
            self._build_process_group(),
            self._build_output_group(),
        ]:
            root.addWidget(widget)
        root.addStretch(1)

        self._apply_style()
        self._load_persisted_values()
        self._apply_mode_defaults(self.mode_box.currentText())

    def _build_source_group(self):
        box = self._group('输入源配置')
        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(12)
        form.addRow('输入模式', self.mode_box)
        form.addRow('模型路径', self.model_row)
        form.addRow('输入源', self.source_row)
        form.addRow('深度模型', self.depth_row)
        return box

    def _build_model_group(self):
        box = self._group('检测与量化参数')
        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(12)
        form.addRow('配置等级', self.preset_combo)
        form.addRow('检测阈值', self.conf_spin)
        form.addRow('有效阈值', self.valid_conf_spin)
        form.addRow('IoU 阈值', self.iou_spin)
        form.addRow('输入尺寸', self.imgsz_spin)
        form.addRow('视频跳帧', self.frame_stride_spin)
        form.addRow('最大帧数', self.max_frames_spin)
        return box

    def _build_process_group(self):
        box = self._group('图像增强与辅助显示')
        layout = QVBoxLayout(box)
        layout.setSpacing(10)
        for widget in [
            self.depth_preview_check,
            self.road_roi_check,
            self.clahe_check,
            self.denoise_check,
            self.augment_check,
            self.tiling_check,
        ]:
            layout.addWidget(widget)
        return box

    def _build_output_group(self):
        box = self._group('输出与资源')
        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(12)
        form.addRow('输出目录', self.output_row)
        form.addRow('字体路径', self.font_row)
        form.addRow('推理设备', self.device_combo)
        form.addRow('', self.save_result_check)
        return box

    def _refresh_device_options(self):
        self.device_combo.clear()
        try:
            import torch

            has_cuda = torch.cuda.is_available()
        except Exception:
            has_cuda = False

        self.device_combo.addItem('CPU（强制）', 'cpu')
        if has_cuda:
            import torch

            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                self.device_combo.addItem(f'GPU {i}（{name}）', str(i))
            self.device_combo.setCurrentIndex(1)
        else:
            self.device_combo.addItem('GPU（不可用，将回退到 CPU）', '0')
            self.device_combo.model().item(1).setEnabled(False)

    def _group(self, title: str):
        box = QGroupBox(title)
        box.setAlignment(Qt.AlignLeft)
        box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        return box

    def _apply_style(self):
        self.setStyleSheet(
            """
            QFrame { background-color: #0b1624; border: 1px solid #173150; border-radius: 18px; }
            QLabel#sectionTitle { color: #eaf2ff; font-size: 18px; font-weight: 800; }
            QLabel#sectionSubtitle { color: #93a8c2; font-size: 12px; }
            QGroupBox {
                color: #dbeafe;
                font-weight: 700;
                border: 1px solid #1a324d;
                border-radius: 14px;
                margin-top: 11px;
                padding: 16px 12px 12px 12px;
                background-color: #0d1928;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 6px; }
            QLabel { color: #d3e2f8; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                min-height: 36px; background-color: #08111d; color: #e2e8f0;
                border: 1px solid #244466; border-radius: 10px; padding: 0 10px;
                selection-background-color: #1f75ff;
            }
            QComboBox QAbstractItemView {
                background-color: #08111d; color: #e2e8f0;
                selection-background-color: #1f75ff; selection-color: #ffffff;
                border: 1px solid #244466;
            }
            QPushButton {
                min-height: 36px; border-radius: 10px; background-color: #123052;
                color: white; padding: 0 14px; border: 1px solid #254a72; font-weight: 700;
            }
            QPushButton:hover { background-color: #18436b; }
            QCheckBox { color: #dbeafe; min-height: 28px; }
            """
        )

    def _load_persisted_values(self) -> None:
        cfg = self.config_service.load() if self.config_service else {}

        self.model_edit.setText(cfg.get('last_model_path', ''))
        self.source_edit.setText(cfg.get('last_source_path', ''))
        self.depth_edit.setText(cfg.get('last_depth_model_path') or cfg.get('depth_model_path', 'depth_model'))
        self.output_edit.setText(cfg.get('last_output_dir', 'runs/predict'))
        self.mode_box.setCurrentText(cfg.get('last_source_mode') or cfg.get('source_mode', '图片'))
        self.imgsz_spin.setValue(int(cfg.get('last_imgsz', cfg.get('imgsz', 960))))
        self.frame_stride_spin.setValue(int(cfg.get('last_frame_stride', cfg.get('frame_stride', 2))))
        self.max_frames_spin.setValue(int(cfg.get('last_max_frames', cfg.get('max_frames', 0))))

    def _apply_preset(self, preset: str) -> None:
        if preset == '小型配置（快速）':
            self.conf_spin.setValue(0.25)
            self.valid_conf_spin.setValue(0.40)
            self.iou_spin.setValue(0.50)
            self.imgsz_spin.setValue(640)
            self.frame_stride_spin.setValue(3)
            self.max_frames_spin.setValue(500)
        elif preset == '中型配置（平衡）':
            self.conf_spin.setValue(0.20)
            self.valid_conf_spin.setValue(0.35)
            self.iou_spin.setValue(0.45)
            self.imgsz_spin.setValue(960)
            self.frame_stride_spin.setValue(2)
            self.max_frames_spin.setValue(0)
        elif preset == '大型配置（精细）':
            self.conf_spin.setValue(0.15)
            self.valid_conf_spin.setValue(0.25)
            self.iou_spin.setValue(0.40)
            self.imgsz_spin.setValue(1280)
            self.frame_stride_spin.setValue(1)
            self.max_frames_spin.setValue(0)

    def _apply_mode_defaults(self, mode: str) -> None:
        if mode == '图片':
            self.conf_spin.setValue(0.10)
            self.valid_conf_spin.setValue(0.10)
            self.frame_stride_spin.setValue(1)
            self.road_roi_check.setChecked(False)
            self.augment_check.setChecked(True)
            self.tiling_check.setChecked(True)
        elif mode == '视频':
            self.conf_spin.setValue(0.12)
            self.valid_conf_spin.setValue(0.20)
            self.frame_stride_spin.setValue(2)
            self.road_roi_check.setChecked(True)
            self.augment_check.setChecked(False)
            self.tiling_check.setChecked(False)
        elif mode == '文件夹':
            self.conf_spin.setValue(0.10)
            self.valid_conf_spin.setValue(0.12)
            self.frame_stride_spin.setValue(1)
            self.road_roi_check.setChecked(False)
            self.augment_check.setChecked(True)
            self.tiling_check.setChecked(True)
        else:
            self.conf_spin.setValue(0.12)
            self.valid_conf_spin.setValue(0.20)
            self.frame_stride_spin.setValue(2)
            self.road_roi_check.setChecked(True)
            self.augment_check.setChecked(False)
            self.tiling_check.setChecked(False)

    def _path_row(self, button_text: str, callback):
        container = QFrame()
        container.setObjectName('pathRow')
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        edit = QLineEdit()
        edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn = QPushButton(button_text)
        btn.setMinimumWidth(74)
        btn.setMaximumWidth(92)
        btn.clicked.connect(callback)
        layout.addWidget(edit, 1)
        layout.addWidget(btn, 0)
        return container, edit

    def _choose_file(self, target: QLineEdit, title: str, file_filter: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, title, '', file_filter)
        if path:
            target.setText(path)

    def _choose_dir(self, target: QLineEdit, title: str) -> None:
        path = QFileDialog.getExistingDirectory(self, title)
        if path:
            target.setText(path)

    def _choose_source(self) -> None:
        mode = self.mode_box.currentText()
        if mode == '摄像头':
            items = ['摄像头 0（默认内置）', '摄像头 1', '摄像头 2', '摄像头 3', '摄像头 4']
            item, ok = QInputDialog.getItem(self, '选择摄像头', '请选择摄像头编号：', items, 0, False)
            if ok:
                self.source_edit.setText(str(items.index(item)))
            return
        if mode == '文件夹':
            self._choose_dir(self.source_edit, '选择图片文件夹')
            return
        file_filter = '媒体文件 (*.png *.jpg *.jpeg *.bmp *.webp *.mp4 *.avi *.mov *.mkv)'
        path, _ = QFileDialog.getOpenFileName(self, '选择输入源', '', file_filter)
        if path:
            self.source_edit.setText(path)

    def to_config(self) -> InferenceConfig:
        return InferenceConfig(
            model_path=self.model_edit.text().strip(),
            source=self.source_edit.text().strip(),
            depth_model_path=self.depth_edit.text().strip() or 'depth_model',
            output_dir=self.output_edit.text().strip() or 'runs/predict',
            font_path=self.font_edit.text().strip() or r'C:\Windows\Fonts\msyh.ttc',
            conf_threshold=float(self.conf_spin.value()),
            valid_conf_threshold=float(self.valid_conf_spin.value()),
            iou_threshold=float(self.iou_spin.value()),
            imgsz=int(self.imgsz_spin.value()),
            frame_stride=int(self.frame_stride_spin.value()),
            max_frames=int(self.max_frames_spin.value()),
            show_depth_preview=self.depth_preview_check.isChecked(),
            show_road_roi=self.road_roi_check.isChecked(),
            enable_clahe=self.clahe_check.isChecked(),
            enable_denoise=self.denoise_check.isChecked(),
            augment=self.augment_check.isChecked(),
            enable_tiling=self.tiling_check.isChecked(),
            save_result=self.save_result_check.isChecked(),
            source_mode=self.mode_box.currentText(),
            device=self.device_combo.currentData() or '',
        )
