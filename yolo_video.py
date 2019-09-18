import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import glob
import os

def detect_img(yolo, input_path, output_path):
    if os.path.isdir(input_path):
        print("Detect folder: ", input_path)
        img_path = glob.glob(os.path.join(input_path, '*'))
    else:
        print("Detect file: ", input_path)
        with open(input_path) as f:
            labels = f.readlines()
        img_path = [line.split()[0] for line in labels]
    for path in img_path:
        try:
            image = Image.open(path)
        except:
            print("Can't open {}".format(path))
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.save(os.path.join(output_path, os.path.basename(path)))
    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_path', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors_path', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes_path', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )
    
    parser.add_argument(
        "--score", type=float, default=0.3,
        help = "Confiden threshold"
    )
    
    parser.add_argument(
        "--iou", type=float, default=0.45,
        help = "IOU threshold"
    )
    
    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        detect_img(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
