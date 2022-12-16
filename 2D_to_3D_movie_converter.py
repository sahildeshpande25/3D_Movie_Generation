import sys
import numpy as np
import cv2
import torch
import imageio
from PIL import Image
from train_stereo_pairs import Left2Right, model_path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Left2Right()
model.load_state_dict(torch.load(model_path.format(9))['model_state_dict'])
model.to(device)

def get_right_image(left_img):
  left = cv2.resize(left_img, (384, 160))
  left = left.astype(np.float32)
  left = np.rollaxis(left, 2, 0)
  left = np.reshape(left, (1, 3, 160, 384))
  left = torch.FloatTensor(left)
  left = left.to(device)

  out = model(left)
  # print('model_out', out.shape)
  right = selectionLayer(out, left).detach().cpu().numpy()
  # print('selection', right.shape)

  left = np.clip(left.cpu().numpy().squeeze().transpose((1,2,0)), 0, 255).astype(np.uint8)
  left = Image.fromarray(cv2.cvtColor(left, cv2.COLOR_BGR2RGB))
  right = np.clip(right.squeeze().transpose((1,2,0)), 0, 255).astype(np.uint8)
  right = Image.fromarray(cv2.cvtColor(right, cv2.COLOR_BGR2RGB))

  # left.save('left.png')
  # right.save('right.png')
  return left, right

def make_anaglyph(left, right):
    width, height = left.size
    leftMap = left.load()
    rightMap = right.load()
    m = [[ 0, 0.7, 0.3, 0, 0, 0, 0, 0, 0 ], [ 0, 0, 0, 0, 1, 0, 0, 0, 1 ]]

    for y in range(0, height):
        for x in range(0, width):
            r1, g1, b1 = leftMap[x, y]
            r2, g2, b2 = rightMap[x, y]
            leftMap[x, y] = (
                int(r1*m[0][0] + g1*m[0][1] + b1*m[0][2] + r2*m[1][0] + g2*m[1][1] + b2*m[1][2]),
                int(r1*m[0][3] + g1*m[0][4] + b1*m[0][5] + r2*m[1][3] + g2*m[1][4] + b2*m[1][5]),
                int(r1*m[0][6] + g1*m[0][7] + b1*m[0][8] + r2*m[1][6] + g2*m[1][7] + b2*m[1][8])
            )

    # left.save('3d.png')
    return left

video_path =  sys.argv[1]
video_path = list(os.path.splitext('video_path'))
ext = video_path[-1]
video_path[-1] = '{}'
video_path.append(ext)
video_path = ''.join(video_path)
vidObj = cv2.VideoCapture(video_path.format(''))
width = int(vidObj.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT))
count = 0
success = 1
frames = []
while success:

    success, image = vidObj.read()
    if success:
      frames.append(image)
      count += 1

frames_3D = []
i = 0
for frame in frames:
  i += 1
  if i%100 == 0:
    print(i)
    # torch.cuda.empty_cache()
  left, right = get_right_image(frame)
  frame_3d = make_anaglyph(left, right)
  frames_3D.append(cv2.resize(np.asarray(frame_3d), (width, height), interpolation=cv2.INTER_LANCZOS4))

imageio.mimwrite(video_path.format('_3D'), frames_3D, fps=24, 
                quality=7, macro_block_size=None)
