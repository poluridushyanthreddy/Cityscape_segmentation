import cv2

def segcolormap(mask):
    h,w=mask.shape
    color_mask=np.zeros((h,w,3),dtype=np.uint8)

    for label, color in SegmentationColorMap.items():
        color_mask[mask==label]=color

    return color_mask

def overlay_segmentation(image, segmap, alpha=0.5):
    seg_color = segcolormap(segmap)
    overlay = cv2.addWeighted(image, 1 - alpha, seg_color, alpha, 0)#final_pixel = image_pixel * (1 - alpha) + seg_pixel * alpha
    return overlay
