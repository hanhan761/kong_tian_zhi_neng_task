"""Launch the installed LabelImg with a JPEG-reader fallback for this PC's Qt."""
from pathlib import Path
from numbers import Real
import sys

from PIL import Image
from PyQt5.QtCore import QLineF, QPointF, QRectF
from PyQt5.QtGui import QImage, QPainter
import labelImg.labelImg as labelimg
import libs.canvas as canvas_module


class CompatiblePainter(QPainter):
    """Select Qt's floating-point overloads for LabelImg's image coordinates."""
    def drawRect(self, *args):
        if len(args) == 4 and all(isinstance(v, Real) for v in args):
            return super().drawRect(QRectF(*map(float, args)))
        return super().drawRect(*args)

    def drawLine(self, *args):
        if len(args) == 4 and all(isinstance(v, Real) for v in args):
            return super().drawLine(QLineF(*map(float, args)))
        return super().drawLine(*args)

    def drawText(self, *args):
        if len(args) == 3 and all(isinstance(v, Real) for v in args[:2]):
            return super().drawText(QPointF(float(args[0]), float(args[1])), args[2])
        return super().drawText(*args)


def configure_qt():
    canvas_module.QPainter = CompatiblePainter
    if getattr(labelimg.MainWindow, '_course_qt_compatible', False):
        return
    original_init = labelimg.MainWindow.__init__

    def compatible_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Wheel zoom/scroll computes floats; QSpinBox/QScrollBar accept ints.
        for widget in [self.zoom_widget] + list(self.scroll_bars.values()):
            original_set = widget.setValue
            widget.setValue = lambda value, setter=original_set: setter(int(round(value)))

    labelimg.MainWindow.__init__ = compatible_init
    labelimg.MainWindow._course_qt_compatible = True


def configure_reader():
    configure_qt()
    original_read = labelimg.read

    def compatible_read(filename, default=None):
        image = original_read(filename, default)
        if isinstance(image, QImage) and not image.isNull():
            return image
        # The installed Qt reports no JPEG plugin. Pillow decodes the actual
        # JPEG into RGBA pixels; QImage receives a detached copy of those pixels.
        with Image.open(filename) as source:
            rgba = source.convert('RGBA')
            pixels = rgba.tobytes()
            return QImage(pixels, rgba.width, rgba.height, rgba.width * 4,
                          QImage.Format_RGBA8888).copy()

    labelimg.read = compatible_read


def prepare_classes(dataset):
    classes = (dataset / 'classes.txt').read_text(encoding='utf-8')
    if classes.splitlines() != ['separation_component', 'flash_smoke']:
        raise ValueError('Unexpected class order; do not open labels with a different class mapping.')
    for split in ['train', 'val', 'test']:
        helper = dataset / 'labels' / split / 'classes.txt'
        if helper.exists() and helper.read_text(encoding='utf-8').splitlines() != classes.splitlines():
            raise ValueError('Existing LabelImg classes.txt disagrees: ' + str(helper))
        if not helper.exists():
            helper.write_text(classes, encoding='utf-8')


if __name__ == '__main__':
    # LabelImg receives images/<split>, classes.txt and labels/<split>.
    # Infer the dataset from that explicit classes file rather than silently
    # preparing a different historical dataset.
    dataset = Path(sys.argv[2]).resolve().parent if len(sys.argv) >= 3 else (
        Path(__file__).resolve().parents[1] / 'separation_dataset_v2_1000')
    prepare_classes(dataset)
    configure_reader()
    sys.exit(labelimg.main())
