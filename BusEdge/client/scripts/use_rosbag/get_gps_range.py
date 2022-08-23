'''
get_gps_range.py
Author: Indu Panigrahi
Description: This script is helpful for obtaining the range of GPS coordinates if you know the image names.
Example command: python get_gps_range.py -b camera3_1641584743_844843503.jpg -e camera3_1641584763_851041244.jpg -i 2022-01-07-14-22-info.pkl
'''

import pickle as pkl
import argparse
import os

def main():
    parser = argparse.ArgumentParser(
        description="Save image file names and GPS info from a rosbag."
    )
    parser.add_argument("-b", "--begin", default="begin.jpg", help="First image in sequence.")
    parser.add_argument("-e", "--end", default="end.jpg", help="Last image in sequence.")
    parser.add_argument("-i", "--input", default="./input.pkl", help="Path to file containing list of dicts containing image and GPS correspondences.")
    args = parser.parse_args()

    first_im = args.begin
    last_im = args.end
    input_file = args.input

    f_in = open(input_file,'rb')
    info_in = pkl.load(f_in)
    f_in.close()

    for im in info_in:
        if im['name'] == first_im:
            print('Begin GPS info:', im['gps'])
        if im['name'] == last_im:
            print('End GPS info:', im['gps'])
            break

    return
        

if __name__ == "__main__":
    main()
