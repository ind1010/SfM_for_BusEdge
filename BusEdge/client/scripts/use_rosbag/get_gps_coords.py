"""
get_gps_coords.py
Author: Indu Panigrahi
Description: Save image file names and GPS info from a rosbag.
Example command: python get_gps_coords.py -i bagname.bag -c 3 -o output.txt
"""

import argparse
import os

import cv2
import rosbag
import yaml
from cv_bridge import CvBridge
from exif import set_gps_location
from sensor_msgs.msg import CompressedImage, NavSatFix
import pickle as pkl


def main():
    parser = argparse.ArgumentParser(
        description="Save image file names and GPS info from a rosbag."
    )
    parser.add_argument("-i", "--input", default="./test.bag", help="Input ROS bag")
    parser.add_argument(
        "-c",
        "--cam-id",
        nargs="+",
        type=int,
        default=[
            3,
        ],
        help="Select camera IDs to extract",
    )
    parser.add_argument("-o", "--output", default="./output.txt", help="Output file")
    args = parser.parse_args()

    bag_file = args.input
    output_filename = args.output

    topics = ["/fix"]
    for cam_id in args.cam_id:
        topics.append("/camera{}/image_raw/compressed".format(cam_id))

    print("Extracting data from {} for topics {}".format(bag_file, topics))

    bag = rosbag.Bag(bag_file, "r")
    info_dict = yaml.load(bag._get_yaml_info())
    print("\nbag_info:\n", info_dict)

    bridge = CvBridge()
    CUR_GPS = NavSatFix()

    lines_to_write = []

    for topic, msg, t in bag.read_messages(topics=topics):
        if "image_raw" in topic:
            time_stamps = "_{:0>10d}_{:0>9d}".format(t.secs, t.nsecs)
            image_filename = topic[1:8] + time_stamps + ".jpg"
            lineDict = {'name':image_filename, 'gps':(CUR_GPS.latitude,CUR_GPS.longitude,CUR_GPS.altitude)}
            lines_to_write.append(lineDict)

        elif "fix" in topic:
            CUR_GPS = msg

    bag.close()

    f = open(output_filename,'wb')
    pkl.dump(lines_to_write,f)
    f.close()

    return


if __name__ == "__main__":
    main()