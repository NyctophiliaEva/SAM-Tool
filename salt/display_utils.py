import cv2
import numpy as np
from pycocotools import mask as coco_mask

class DisplayUtils:
    def __init__(self):
        self.transparency = 0.3
        # 让边框与文字更淡
        self.box_width = 1
        self.label_font_scale = 0.7  # 原先约 1.5
        self.label_thickness = 1     # 原先 5
        self.label_bg_alpha = 0.25   # 文字背景半透明强度（0~1）

    def increase_transparency(self):
        self.transparency = min(1.0, self.transparency + 0.05)
    
    def decrease_transparency(self):
        self.transparency = max(0.0, self.transparency - 0.05)

    def overlay_mask_on_image(self, image, mask, color=(0, 0, 255)):
        gray_mask = mask.astype(np.uint8) * 255
        gray_mask = cv2.merge([gray_mask, gray_mask, gray_mask])
        color_mask = cv2.bitwise_and(gray_mask, color)
        masked_image = cv2.bitwise_and(image.copy(), color_mask)
        overlay_on_masked_image = cv2.addWeighted(
            masked_image, self.transparency, color_mask, 1 - self.transparency, 0
        )
        background = cv2.bitwise_and(image.copy(), cv2.bitwise_not(color_mask))
        image = cv2.add(background, overlay_on_masked_image)
        return image

    def __convert_ann_to_mask(self, ann, height, width):
        mask = np.zeros((height, width), dtype=np.uint8)
        poly = ann["segmentation"]
        rles = coco_mask.frPyObjects(poly, height, width)
        rle = coco_mask.merge(rles)
        mask_instance = coco_mask.decode(rle)
        mask_instance = np.logical_not(mask_instance)
        mask = np.logical_or(mask, mask_instance)
        mask = np.logical_not(mask)
        return mask

    def draw_box_on_image(self, image, categories, ann, color, draw_label=True):
        x, y, w, h = ann["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)
        # 边框更细，降低突兀感
        image = cv2.rectangle(image, (x, y), (x + w, y + h), color, self.box_width)

        if draw_label:
            text = '{} {}'.format(ann["id"],categories[ann["category_id"]])
            txt_color = (0, 0, 0) if np.mean(color) > 127 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            # 减小字号、减小厚度
            txt_size, _ = cv2.getTextSize(text, font, self.label_font_scale, self.label_thickness)
            # 绘制半透明背景矩形
            overlay = image.copy()
            x2 = x + txt_size[0] + 6
            y2 = y + int(1.6 * txt_size[1])
            cv2.rectangle(overlay, (x, y + 1), (x2, y2), color, -1)
            image = cv2.addWeighted(overlay, self.label_bg_alpha, image, 1 - self.label_bg_alpha, 0)
            # 绘制更小更细的文字
            cv2.putText(image, text, (x + 3, y + int(1.1 * txt_size[1])), font, self.label_font_scale, txt_color, thickness=self.label_thickness, lineType=cv2.LINE_AA)
        return image

    def draw_annotations(self, image, categories, annotations, colors, visible_label_ids=None):
        show_all = visible_label_ids is None
        for ann, color in zip(annotations, colors):
            draw_label = show_all or (ann.get("id") in visible_label_ids)
            image = self.draw_box_on_image(image, categories, ann, color, draw_label=draw_label)
            mask = self.__convert_ann_to_mask(ann, image.shape[0], image.shape[1])
            image = self.overlay_mask_on_image(image, mask, color)
        return image

    def draw_points(
        self, image, points, labels, colors={1: (0, 255, 0), 0: (0, 0, 255)}, radius=5
    ):
        for i in range(points.shape[0]):
            point = points[i, :]
            label = labels[i]
            color = colors[label]
            image = cv2.circle(image, tuple(point), radius, color, -1)
        return image
