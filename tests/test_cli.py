# -*- coding: utf-8 -*-
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from click.testing import CliRunner
from PIL import Image

from helpers import temp_output

from kraken.kraken import cli

thisfile = Path(__file__).resolve().parent
resources = thisfile / 'resources'


class TestCLI(unittest.TestCase):
    """
    Testing the kraken CLI (binarization subcommand).
    """

    def setUp(self):
        self.runner = CliRunner()
        self.color_img = resources / 'input.webp'
        self.bw_img = resources / 'input_bw.png'

    def test_binarize_color(self):
        """
        Tests binarization of color images.
        """
        with tempfile.NamedTemporaryFile() as fp:
            result = self.runner.invoke(cli, ['-i', self.color_img, fp.name, 'binarize'])
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(tuple(map(lambda x: x[1], Image.open(fp).getcolors())), (0, 255))

    def test_binarize_bw(self):
        """
        Tests binarization of b/w images.
        """
        with tempfile.NamedTemporaryFile() as fp:
            result = self.runner.invoke(cli, ['-i', self.bw_img, fp.name, 'binarize'])
            self.assertEqual(result.exit_code, 0)
            bw = np.array(Image.open(self.bw_img))
            new = np.array(Image.open(fp.name))
            self.assertTrue(np.all(bw == new))

    def test_segment_color(self):
        """
        Tests that segmentation is aborted when given color image.
        """
        with tempfile.NamedTemporaryFile() as fp:
            result = self.runner.invoke(cli, ['-r', '-i', self.color_img, fp.name, 'segment'])
            self.assertEqual(result.exit_code, 1)


class TestCLIPDFInput(unittest.TestCase):
    """
    Tests for `-f pdf` multi-page PDF extraction (pypdfium2-based).
    """

    def setUp(self):
        self.runner = CliRunner()

    def test_pdf_extraction_multi_page(self):
        """
        Tests that each page of a multi-page PDF is extracted and processed.
        """
        with self.runner.isolated_filesystem():
            from PIL import ImageDraw
            page1 = Image.new('RGB', (64, 64), 'white')
            ImageDraw.Draw(page1).rectangle((8, 8, 56, 56), fill='black')
            page2 = Image.new('RGB', (64, 64), 'white')
            ImageDraw.Draw(page2).ellipse((8, 8, 56, 56), fill='black')
            page1.save('doc.pdf', save_all=True, append_images=[page2])
            result = self.runner.invoke(cli, ['-f', 'pdf', '-I', 'doc.pdf', '-o', '.png', 'binarize'])
            self.assertEqual(result.exit_code, 0, msg=result.output)
            outputs = sorted(Path('.').glob('doc.pdf_*.png'))
            self.assertEqual(len(outputs), 2)
            # pages are rasterized at 300dpi; Pillow saves PDFs at 72dpi by
            # default, so a 64px page comes back scaled by 300/72
            expected_size = round(64 * 300 / 72)
            for out in outputs:
                w, h = Image.open(out).size
                self.assertAlmostEqual(w, expected_size, delta=1)
                self.assertAlmostEqual(h, expected_size, delta=1)

    def test_pdf_extraction_invalid_file_skipped(self):
        """
        Tests that a non-PDF file passed with `-f pdf` is skipped with a
        warning rather than raising.
        """
        with self.runner.isolated_filesystem():
            Image.new('RGB', (32, 32), 'white').save('notapdf.pdf')
            result = self.runner.invoke(cli, ['-f', 'pdf', '-I', 'notapdf.pdf', '-o', '.png', 'binarize'])
            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertEqual(len(list(Path('.').glob('notapdf.pdf_*.png'))), 0)


class TestCLISegmentation(unittest.TestCase):
    """
    Integration tests for the kraken CLI segment subcommand (neural baseline
    segmentation).
    """

    def setUp(self):
        self.runner = CliRunner()
        self.bw_img = resources / 'input.webp'

    def test_segment_baseline_native_output(self):
        """
        Tests baseline segmentation with native JSON output.
        """
        with temp_output(suffix='.json') as fname:
            result = self.runner.invoke(cli, ['-i', str(self.bw_img), fname, 'segment', '-bl'])
            self.assertEqual(result.exit_code, 0, msg=result.output)
            with open(fname, 'r') as f:
                data = json.load(f)
            self.assertIn('type', data)
            self.assertEqual(data['type'], 'baselines')
            self.assertIn('lines', data)
            self.assertGreater(len(data['lines']), 0)

    def test_segment_baseline_alto_output(self):
        """
        Tests baseline segmentation with ALTO XML serialization.
        """
        with temp_output(suffix='.xml') as fname:
            result = self.runner.invoke(cli, ['-a', '-i', str(self.bw_img), fname, 'segment', '-bl'])
            self.assertEqual(result.exit_code, 0, msg=result.output)
            with open(fname, 'r') as f:
                content = f.read()
            self.assertIn('alto', content.lower())
            self.assertNotIn('<fileName>None</fileName>', content)
            self.assertIn(self.bw_img.name, content)

    def test_segment_baseline_text_direction(self):
        """
        Tests that the text direction option is accepted.
        """
        with temp_output(suffix='.json') as fname:
            result = self.runner.invoke(cli, ['-i', str(self.bw_img), fname,
                                              'segment', '-bl', '-d', 'horizontal-rl'])
            self.assertEqual(result.exit_code, 0, msg=result.output)
            with open(fname, 'r') as f:
                data = json.load(f)
            self.assertEqual(data['text_direction'], 'horizontal-rl')


class TestCLIRecognition(unittest.TestCase):
    """
    Integration tests for the kraken CLI ocr subcommand.
    """

    def setUp(self):
        self.runner = CliRunner()
        self.bw_img = resources / 'bw.png'
        self.model = resources / 'overfit.mlmodel'

    def test_ocr_no_segmentation_mode(self):
        """
        Tests recognition in no-segmentation mode (whole image as one line).
        """
        with temp_output(suffix='.txt') as fname:
            result = self.runner.invoke(cli, ['-i', str(self.bw_img), fname,
                                              'ocr', '-m', str(self.model), '-s'])
            self.assertEqual(result.exit_code, 0, msg=result.output)
            with open(fname, 'r') as f:
                content = f.read()
            self.assertGreater(len(content.strip()), 0)

    def test_ocr_pipeline_segment_then_ocr(self):
        """
        Tests the chained segment -> ocr pipeline.
        """
        with temp_output(suffix='.txt') as fname:
            result = self.runner.invoke(cli, ['-i', str(self.bw_img), fname,
                                              'segment', '-bl',
                                              'ocr', '-m', str(self.model)])
            self.assertEqual(result.exit_code, 0, msg=result.output)
            with open(fname, 'r') as f:
                content = f.read()
            self.assertGreater(len(content.strip()), 0)

    def test_ocr_output_formats(self):
        """
        Tests recognition output serialization for each format flag.
        """
        cases = [('alto', '-a', '.xml', lambda c: 'alto' in c.lower()),
                 ('hocr', '-h', '.html', lambda c: 'ocr_page' in c),
                 ('pagexml', '-x', '.xml', lambda c: 'PcGts' in c)]
        for name, flag, suffix, marker in cases:
            with self.subTest(name):
                with temp_output(suffix=suffix) as fname:
                    result = self.runner.invoke(cli, [flag, '-i', str(self.bw_img), fname,
                                                      'segment', '-bl',
                                                      'ocr', '-m', str(self.model)])
                    self.assertEqual(result.exit_code, 0, msg=result.output)
                    with open(fname, 'r') as f:
                        content = f.read()
                    self.assertTrue(marker(content), msg=f'format marker missing in {name} output')

    def test_ocr_xml_input_linetype(self):
        """
        Tests recognition on XML input with the linetype derived from the
        model and with an explicit override.
        """
        xml = resources / '170025120000003,0074-lite.xml'
        for name, extra in (('auto', []), ('explicit', ['--linetype', 'baselines'])):
            with self.subTest(name):
                with temp_output(suffix='.txt') as fname:
                    result = self.runner.invoke(cli, ['-f', 'xml', '-i', str(xml), fname,
                                                      'ocr', '-m', str(self.model)] + extra)
                    self.assertEqual(result.exit_code, 0, msg=result.output)
                    with open(fname, 'r') as f:
                        content = f.read()
                    self.assertGreater(len(content.strip()), 0)

    def test_ocr_missing_model_fails(self):
        """
        Tests that ocr fails gracefully when model file doesn't exist.
        """
        with temp_output(suffix='.txt') as fname:
            result = self.runner.invoke(cli, ['-i', str(self.bw_img), fname,
                                              'ocr', '-m', 'nonexistent_model.mlmodel', '-s'])
            self.assertNotEqual(result.exit_code, 0)
