from __future__ import print_function
import cv2
import numpy as np
 
 
MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15
 
 
def alignImages(im1, im2):
 
  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
   
  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_FEATURES)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
   
  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)
   
  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)
 
  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]
 
  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  cv2.imwrite("matches.jpg", imMatches)
   
  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)
 
  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
   
  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
 
  # Use homography
  height, width, channels = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))
   
  return im1Reg, h
 
 
if __name__ == '__main__':
   
  # Read reference image
  refFilename = "C:\\Users\\Dhaarani\\Downloads\\Image_Processing_Challenge (2)\\augmented_new\\Original\\original.jpg"
  print("Reading reference image : ", refFilename)
  imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
 
  # Read image to be aligned
  imFilename = "C:\\Users\\Dhaarani\\Downloads\\Image_Processing_Challenge (2)\\augmented_new\\Test_images\\offset_scale.jpg"
  print("Reading image to align : ", imFilename);  
  im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
   
  print("Aligning images ...")
  # Registered image will be resotred in imReg. 
  # The estimated homography will be stored in h. 
  imReg, h = alignImages(im, imReference)
   
  # Write aligned image to disk. 
  outFilename = "C:\\Users\\Dhaarani\\Downloads\\Image_Processing_Challenge (2)\\augmented_new\\Test_images\\C_01.jpg"
  print("Saving aligned image : ", outFilename); 
  cv2.imwrite(outFilename, imReg)
 
  # Print estimated homography
  print("Estimated homography : \n",  h)

    
  imFilename = "C:\\Users\\Dhaarani\\Downloads\\Image_Processing_Challenge (2)\\augmented_new\\Test_images\\offset_scale10_rot45.jpg"
  print("Reading image to align : ", imFilename);  
  im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
   
  print("Aligning images ...")
  # Registered image will be resotred in imReg. 
  # The estimated homography will be stored in h. 
  imReg, h = alignImages(im, imReference)
   
  # Write aligned image to disk. 
  outFilename = "C:\\Users\\Dhaarani\\Downloads\\Image_Processing_Challenge (2)\\augmented_new\\Test_images\\C_02.jpg"
  print("Saving aligned image : ", outFilename); 
  cv2.imwrite(outFilename, imReg)
    
  imFilename = "C:\\Users\\Dhaarani\\Downloads\\Image_Processing_Challenge (2)\\augmented_new\\Test_images\\ofset_down_scale.jpg"
  print("Reading image to align : ", imFilename);  
  im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
   
  print("Aligning images ...")
  # Registered image will be resotred in imReg. 
  # The estimated homography will be stored in h. 
  imReg, h = alignImages(im, imReference)
   
  # Write aligned image to disk. 
  outFilename = "C:\\Users\\Dhaarani\\Downloads\\Image_Processing_Challenge (2)\\augmented_new\\Test_images\\C_03.jpg"
  print("Saving aligned image : ", outFilename); 
  cv2.imwrite(outFilename, imReg)
    
  imFilename = "C:\\Users\\Dhaarani\\Downloads\\Image_Processing_Challenge (2)\\augmented_new\\Test_images\\rot25_offset_scale10.jpg"
  print("Reading image to align : ", imFilename);  
  im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
   
  print("Aligning images ...")
  # Registered image will be resotred in imReg. 
  # The estimated homography will be stored in h. 
  imReg, h = alignImages(im, imReference)
   
  # Write aligned image to disk. 
  outFilename = "C:\\Users\\Dhaarani\\Downloads\\Image_Processing_Challenge (2)\\augmented_new\\Test_images\\C_04.jpg"
  print("Saving aligned image : ", outFilename); 
  cv2.imwrite(outFilename, imReg)
    
  imFilename = "C:\\Users\\Dhaarani\\Downloads\\Image_Processing_Challenge (2)\\augmented_new\\Test_images\\rot180_scale20.jpg"
  print("Reading image to align : ", imFilename);  
  im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
   
  print("Aligning images ...")
  # Registered image will be resotred in imReg. 
  # The estimated homography will be stored in h. 
  imReg, h = alignImages(im, imReference)
   
  # Write aligned image to disk. 
  outFilename = "C:\\Users\\Dhaarani\\Downloads\\Image_Processing_Challenge (2)\\augmented_new\\Test_images\\C_05.jpg"
  print("Saving aligned image : ", outFilename); 
  cv2.imwrite(outFilename, imReg)
    
  imFilename = "C:\\Users\\Dhaarani\\Downloads\\Image_Processing_Challenge (2)\\augmented_new\\Test_images\\rot180_trans.jpg"
  print("Reading image to align : ", imFilename);  
  im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
   
  print("Aligning images ...")
  # Registered image will be resotred in imReg. 
  # The estimated homography will be stored in h. 
  imReg, h = alignImages(im, imReference)
   
  # Write aligned image to disk. 
  outFilename = "C:\\Users\\Dhaarani\\Downloads\\Image_Processing_Challenge (2)\\augmented_new\\Test_images\\C_06.jpg"
  print("Saving aligned image : ", outFilename); 
  cv2.imwrite(outFilename, imReg)
    
  imFilename = "C:\\Users\\Dhaarani\\Downloads\\Image_Processing_Challenge (2)\\augmented_new\\Test_images\\rot300.jpg"
  print("Reading image to align : ", imFilename);  
  im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
   
  print("Aligning images ...")
  # Registered image will be resotred in imReg. 
  # The estimated homography will be stored in h. 
  imReg, h = alignImages(im, imReference)
   
  # Write aligned image to disk. 
  outFilename = "C:\\Users\\Dhaarani\\Downloads\\Image_Processing_Challenge (2)\\augmented_new\\Test_images\\C_07.jpg"
  print("Saving aligned image : ", outFilename); 
  cv2.imwrite(outFilename, imReg)
    
  imFilename = "C:\\Users\\Dhaarani\\Downloads\\Image_Processing_Challenge (2)\\augmented_new\\Test_images\\rotation10_transformation.jpg"
  print("Reading image to align : ", imFilename);  
  im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
   
  print("Aligning images ...")
  # Registered image will be resotred in imReg. 
  # The estimated homography will be stored in h. 
  imReg, h = alignImages(im, imReference)
   
  # Write aligned image to disk. 
  outFilename = "C:\\Users\\Dhaarani\\Downloads\\Image_Processing_Challenge (2)\\augmented_new\\Test_images\\C_08.jpg"
  print("Saving aligned image : ", outFilename); 
  cv2.imwrite(outFilename, imReg)
    
  imFilename = "C:\\Users\\Dhaarani\\Downloads\\Image_Processing_Challenge (2)\\augmented_new\\Test_images\\scale10_rot45.jpg"
  print("Reading image to align : ", imFilename);  
  im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
   
  print("Aligning images ...")
  # Registered image will be resotred in imReg. 
  # The estimated homography will be stored in h. 
  imReg, h = alignImages(im, imReference)
   
  # Write aligned image to disk. 
  outFilename = "C:\\Users\\Dhaarani\\Downloads\\Image_Processing_Challenge (2)\\augmented_new\\Test_images\\C_09.jpg"
  print("Saving aligned image : ", outFilename); 
  cv2.imwrite(outFilename, imReg)
    
  imFilename = "C:\\Users\\Dhaarani\\Downloads\\Image_Processing_Challenge (2)\\augmented_new\\Test_images\\trans_offset_scale.jpg"
  print("Reading image to align : ", imFilename);  
  im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
   
  print("Aligning images ...")
  # Registered image will be resotred in imReg. 
  # The estimated homography will be stored in h. 
  imReg, h = alignImages(im, imReference)
   
  # Write aligned image to disk. 
  outFilename = "C:\\Users\\Dhaarani\\Downloads\\Image_Processing_Challenge (2)\\augmented_new\\Test_images\\C_10.jpg"
  print("Saving aligned image : ", outFilename); 
  cv2.imwrite(outFilename, imReg)