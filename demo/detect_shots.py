import os.path as osp

from movienet.tools import ShotDetector

if __name__ == '__main__':
    sdt = ShotDetector(
        print_list=True,
        save_keyf=True,
        save_keyf_txt=True,
        split_video=False,
        save_stat_csv=True
    )
        # begin_frame=0,
        # end_frame=2000)

    video_path = 'data/titanic.mp4'
    out_dir = osp.join('data', 'titanic')
    sdt.shotdetect(video_path, out_dir)
