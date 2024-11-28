import cv2
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

OUTPUT_CSV_ROOT = '../soft_ensemble/KFold_weight_25.csv'

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]


# RLE로 인코딩된 결과를 mask map으로 복원합니다.
def decode_rle_to_mask(rle, height, width):
    try:
        s = rle.split()
    except AttributeError:
        print(rle)

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)


# mask map으로 나오는 인퍼런스 결과를 RLE로 인코딩 합니다.
def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length 
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def method_one(mask, kernel):
    return cv2.erode(mask, kernel)

def method_two(mask, kernel):
    return cv2.dilate(mask, kernel)

def method_three(mask, kernel):
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

def method_four(mask, kernel):
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

def method_five(mask, kernel):
    tmp = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(tmp, cv2.MORPH_CLOSE, kernel)

def method_six(mask, kernel):
    tmp = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(tmp, cv2.MORPH_OPEN, kernel)

def print_setting(args):
    print("\n======== Print Setting  ========")
    print(f"불러오는 csv 파일 경로 : {args.csv}")
    print(f"사용하는 Kernel Sahpe : cv2.MORPH_{args.kshape.upper()}")
    print(f"사용하는 Kernel Size : {(args.ksize, args.ksize)}")
    method_text = {
        1 : "Only Erosion",
        2 : "Only Dilatation",
        3 : "Opening (Erosion -> Dilatation)",
        4 : "Closing (Dilatation -> Erosion)",
        5 : "Opening -> Closing",
        6 : "Closing -> Opening",
    }
    print(f"사용하는 Method : {args.method}번 !! {method_text[args.method]}")
    print(f"저장될 Output File Name : {args.output}")

def apply_morphology(args):
    print_setting(args)

    df = pd.read_csv(args.csv)

    method_dict = {
        1 : method_one,
        2 : method_two,
        3 : method_three,
        4 : method_four,
        5 : method_five,
        6 : method_six,
    }

    kernel = cv2.getStructuringElement(getattr(cv2, "MORPH_" + args.kshape.upper()), (args.ksize, args.ksize))
    morphology_results = []

    print("\n======== Morphology Calculating Start ========")
    for image_fname, group in tqdm(df.groupby('image_name'), desc="Morphology Calculate", total=len(df['image_name'].unique())):
        for rle in group['rle']:
            mask = decode_rle_to_mask(rle, height=2048, width=2048)
            mask = method_dict[args.method](mask, kernel)
            new_rle = encode_mask_to_rle(mask)
            morphology_results.append(new_rle)
            
    print("\n======== Save Output ========")
    try:
        df['rle'] = morphology_results
        df.to_csv(args.output, index=False)
    except Exception as e:
        print(f"{args.output}을 생성하는데 실패하였습니다.. : {e}")
        raise

    print(f"{args.output} 생성 완료")

if __name__=="__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("csv", type=str, help="Path to the csv file to use")
    parser.add_argument("--method", 
                        type=int, 
                        choices=[1, 2, 3, 4, 5, 6], 
                        default=6,
                        help="Select the method:\n1: Use method Erosion\n2: Use method Dilatation\n3: Use method Opening\n4: Use method Closing\n5: Use method Opening, Closing\n6: Use method Closing, Opening")

    parser.add_argument("--ksize", type=int, default=3, help="kernel size")
    parser.add_argument("--kshape", type=str, choices=["rect", "cross", "ellipse"], default="rect", help="kernel shape")
    parser.add_argument("--output", type=str, default="output.csv")
    args = parser.parse_args()

    if args.ksize % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")

    apply_morphology(args)