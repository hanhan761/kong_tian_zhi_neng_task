"""Check LabelImg can open a real image and its existing YOLO boxes, without saving."""
import argparse
import json
import hashlib
import os
from pathlib import Path
import sys

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
ROOT = Path(__file__).resolve().parents[1]
from PyQt5.QtCore import QPointF, Qt, QTimer, QT_VERSION_STR, PYQT_VERSION_STR
from labelImg.labelImg import MainWindow, QApplication
from libs.shape import Shape
from launch_labelimg import configure_reader, prepare_classes

app = QApplication([])
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset', default='separation_dataset_v2_1000')
parser.add_argument('--sample', default='sts117_srb_050')
args = parser.parse_args()
data = ROOT / args.dataset
sample = data / ('images/train/' + args.sample + '.jpg')
label = data / ('labels/train/' + args.sample + '.txt')
expected_shapes = len([line for line in label.read_text(encoding='utf-8').splitlines() if line.strip()])
before_hash = hashlib.sha256(label.read_bytes()).hexdigest()
prepare_classes(data)
configure_reader()
# A failed image load should produce an exception in this check, not a modal
# GUI dialog that blocks an unattended offscreen environment check.
def fail_dialog(self, title, message):
    raise RuntimeError(title + ': ' + message)
MainWindow.error_message = fail_dialog
window = MainWindow(str(sample), str(data / 'classes.txt'), str(data / 'labels/train'))
window.show()
result = {}
errors = []
sys.excepthook = lambda kind, value, tb: errors.append(str(value))


def exercise_drawing():
    canvas = window.canvas
    canvas.set_editing(False)
    canvas.current = Shape()
    canvas.current.add_point(QPointF(50.5, 60.5))
    canvas.line.points = [QPointF(50.5, 60.5), QPointF(130.5, 140.5)]
    canvas.prev_point = QPointF(130.5, 140.5)
    window.set_zoom(100.5)
    window.scroll_request(120, Qt.Vertical)
    window.zoom_request(120)
    # Force the preview rectangle, crosshair and floating label text branches.
    for shape in canvas.shapes:
        shape.paint_label = True
    canvas.repaint()
    window.grab()
    canvas.current = None
    canvas.line.points = []
    canvas.set_editing(True)
    result['drawing_preview_and_wheel_exercised'] = True
    QTimer.singleShot(300, inspect)


def inspect():
    result.update(image_loaded=not window.image.isNull(), shapes_loaded=len(window.canvas.shapes),
                  classes=list(window.label_hist), expected_shapes=expected_shapes,
                  save_directory=window.default_save_dir,
                  format=str(window.label_file_format), sample=str(sample),
                  qt=QT_VERSION_STR, pyqt=PYQT_VERSION_STR,
                  qt_callback_errors=errors,
                  source_label_unchanged=hashlib.sha256(label.read_bytes()).hexdigest() == before_hash,
                  scope='Read-only real-image loading plus synthetic drawing-preview/zoom/scroll; no manual annotation or saving performed.')
    print(json.dumps(result, ensure_ascii=True), flush=True)
    report = ROOT / 'reports/04_交付汇总' / ('labelimg_readonly_check_' + data.name + '.json')
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    window.grab().save(str(report.parent / ('LabelImg_读取检查_' + data.name + '.png')))
    app.quit()


QTimer.singleShot(1500, exercise_drawing)
app.exec_()
if (not result.get('image_loaded') or result.get('shapes_loaded') != expected_shapes
        or errors or not result.get('source_label_unchanged')):
    sys.exit(1)
