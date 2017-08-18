# coding:utf-8
from __future__ import print_function
from __future__ import absolute_import

import argparse
import numpy as np
import os

from PIL import Image, ImageDraw, ImageFont

import collections


def load_charset(char_dir):
    with open(char_dir, 'r') as f:
        charset = f.readlines()
        charset = [char.strip() for char in charset]
        return charset


def draw_single_char(ch, font, canvas_size, x_offset, y_offset):
    img = Image.new("L", (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, 0, font=font)
    return img


def draw_example(ch, src_font, dst_font, canvas_size, x_offset, y_offset, filter_hashes):
    dst_img = draw_single_char(ch, dst_font, canvas_size, x_offset, y_offset)
    # check the filter example in the hashes or not
    dst_hash = hash(dst_img.tobytes())
    if dst_hash in filter_hashes:
        return None
    src_img = draw_single_char(ch, src_font, canvas_size, x_offset, y_offset)
    example_img = Image.new("L", (canvas_size * 2, canvas_size), 255)
    example_img.paste(dst_img, (0, 0))
    example_img.paste(src_img, (canvas_size, 0))
    return example_img


def filter_recurring_hash(charset, font, canvas_size, x_offset, y_offset):
    _charset = charset[:]
    np.random.shuffle(_charset)
    sample = _charset[:2000]
    hash_count = collections.defaultdict(int)
    for c in sample:
        img = draw_single_char(c, font, canvas_size, x_offset, y_offset)
        hash_count[hash(img.tobytes())] += 1
    recurring_hashes = filter(lambda d: d[1] > 2, hash_count.items())
    return [rh[0] for rh in recurring_hashes]


def font2image(src, dst, charset, char_size, canvas_size, x_offset, y_offset, sample_dir, filter_by_hash=True):
    src_font = ImageFont.truetype(src, size=char_size)
    dst_font = ImageFont.truetype(dst, size=char_size)

    filter_hashes = set()
    if filter_by_hash:
        filter_hashes = set(filter_recurring_hash(charset, dst_font, canvas_size, x_offset, y_offset))
        print("filter hashes -> %s" % (",".join([str(h) for h in filter_hashes])))

    count = 0
    for c in charset:
        e = draw_example(c, src_font, dst_font, canvas_size, x_offset, y_offset, filter_hashes)
        if e:
            e.save(os.path.join(sample_dir, "%04d.jpg" % count))
            count += 1
            if count % 100 == 0:
                print("processed %d chars" % count)


parser = argparse.ArgumentParser(description="Convert font to images")
parser.add_argument("--char_dir", dest="char_dir", required=True, help="path of the characters")
parser.add_argument("--src_font", dest="src_font", required=True, help="path of the source font")
parser.add_argument("--dst_font", dest="dst_font", required=True, help="path of the target font")
parser.add_argument("--shuffle", dest="shuffle", type=int, default=0, help="shuffle a charset before processing")
parser.add_argument("--char_size", dest="char_size", type=int, default=256, help="character size")
parser.add_argument("--canvas_size", dest="canvas_size", type=int, default=256, help="canvas size")
parser.add_argument("--x_offset", dest="x_offset", type=int, default=0, help="x offset")
parser.add_argument("--y_offset", dest="y_offset", type=int, default=0, help="y offset")
parser.add_argument("--sample_dir", dest="sample_dir", help="directory to save sample")


args = parser.parse_args()

if __name__ == "__main__":
    if not os.path.exists(args.sample_dir):
        os.mkdir(args.sample_dir)

    charset = load_charset(args.char_dir)

    if args.shuffle:
        np.random.shuffle(charset)
    font2image(args.src_font, args.dst_font, charset, args.char_size, args.canvas_size, args.x_offset, args.y_offset,
               args.sample_dir)

