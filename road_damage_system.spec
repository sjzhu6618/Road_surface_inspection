# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec — 方案A：只打包 GUI 主程序，predict.py 作为外部文件

import sys
from pathlib import Path

block_cipher = None

ROOT = Path(SPECPATH)

a = Analysis(
    [str(ROOT / 'launch_desktop.py')],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[
        # 将 predict.py 复制到输出目录根
        (str(ROOT / 'predict.py'), '.'),
        # configs 目录（运行时生成，先建空目录占位）
        # runs/predict 目录不打包，运行时自动创建
    ],
    hiddenimports=[
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        'PySide6.QtNetwork',
        'matplotlib',
        'matplotlib.backends.backend_qtagg',
        'matplotlib.figure',
        'scipy',
        'PIL',
        'PIL.Image',
        'cv2',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 排除推理依赖（由外部 Python 环境提供）
        'torch',
        'torchvision',
        'ultralytics',
        'transformers',
        'cv2',
        'sklearn',
        'tensorflow',
        'keras',
        'jupyter',
        'notebook',
        'IPython',
        'pytest',
        'setuptools',
        'pkg_resources',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='road_damage_system',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # 不显示控制台窗口
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='road_damage_system',
)
