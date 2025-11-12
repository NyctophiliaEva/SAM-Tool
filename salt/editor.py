import os, copy
import numpy as np

from salt.onnx_model import OnnxModel
from salt.dataset_explorer import DatasetExplorer
from salt.display_utils import DisplayUtils

class CurrentCapturedInputs:
    def __init__(self):
        self.input_point = np.array([])
        self.input_label = np.array([])
        self.low_res_logits = None
        self.curr_mask = None

    def reset_inputs(self):
        self.input_point = np.array([])
        self.input_label = np.array([])
        self.low_res_logits = None
        self.curr_mask = None

    def set_mask(self, mask):
        self.curr_mask = mask

    def add_input_click(self, input_point, input_label):
        if len(self.input_point) == 0:
            self.input_point = np.array([input_point])
        else:
            self.input_point = np.vstack([self.input_point, np.array([input_point])])
        self.input_label = np.append(self.input_label, input_label)

    def set_low_res_logits(self, low_res_logits):
        self.low_res_logits = low_res_logits


class Editor:
    def __init__(self, onnx_model_path, dataset_path, categories=None, coco_json_path=None):
        self.dataset_path = dataset_path
        self.coco_json_path = coco_json_path
        self.onnx_model_path = onnx_model_path
        self.onnx_helper = OnnxModel(self.onnx_model_path)
        if categories is None and not os.path.exists(coco_json_path):
            raise ValueError("categories must be provided if coco_json_path is None")
        if self.coco_json_path is None:
            self.coco_json_path = os.path.join(self.dataset_path, "annotations.json")
        self.dataset_explorer = DatasetExplorer(
            self.dataset_path, categories=categories, coco_json_path=self.coco_json_path
        )
        self.curr_inputs = CurrentCapturedInputs()
        self.categories = self.dataset_explorer.get_categories()
        self.image_id = 0
        self.category_id = 0
        self.show_other_anns = True
        (
            self.image,
            self.image_bgr,
            self.image_embedding,
        ) = self.dataset_explorer.get_image_data(self.image_id)
        self.display = self.image_bgr.copy()
        self.du = DisplayUtils()
        self.hover_ann_ids = set()
        self.hover_mode_enabled = True  # 是否启用“仅悬停显示文字”模式
        self.reset()

    # ---- 图片导航与信息接口 ----
    def get_image_count(self):
        return self.dataset_explorer.get_num_images()

    def get_current_index(self):
        return self.image_id

    def get_current_filename(self):
        image_name = self.dataset_explorer.coco_json["images"][self.image_id]["file_name"]
        return os.path.basename(image_name)

    def add_click(self, new_pt, new_label):
        self.curr_inputs.add_input_click(new_pt, new_label)
        masks, low_res_logits = self.onnx_helper.call(
            self.image,
            self.image_embedding,
            self.curr_inputs.input_point,
            self.curr_inputs.input_label,
            low_res_logits=self.curr_inputs.low_res_logits,
        )
        self.display = self.image_bgr.copy()
        self.draw_known_annotations()
        self.display = self.du.draw_points(
            self.display, self.curr_inputs.input_point, self.curr_inputs.input_label
        )
        self.display = self.du.overlay_mask_on_image(self.display, masks[0, 0, :, :])
        self.curr_inputs.set_mask(masks[0, 0, :, :])
        self.curr_inputs.set_low_res_logits(low_res_logits)

    def draw_known_annotations(self):
        anns, colors = self.dataset_explorer.get_annotations(
            self.image_id, return_colors=True
        )
        if self.hover_mode_enabled:
            # 仅在悬停时显示文字
            self.display = self.du.draw_annotations(
                self.display, self.categories, anns, colors, visible_label_ids=self.hover_ann_ids
            )
        else:
            # 显示全部标签
            self.display = self.du.draw_annotations(
                self.display, self.categories, anns, colors, visible_label_ids=None
            )

    def reset(self, hard=True):
        self.curr_inputs.reset_inputs()
        self.display = self.image_bgr.copy()
        # 清除悬停状态
        self.hover_ann_ids = set()
        if self.show_other_anns:
            self.draw_known_annotations()

    def toggle(self):
        self.show_other_anns = not self.show_other_anns
        self.reset()

    def toggle_hover_mode(self):
        self.hover_mode_enabled = not self.hover_mode_enabled
        # 切换模式后需要强制刷新显示（重新绘制标签）
        self.display = self.image_bgr.copy()
        if self.show_other_anns:
            self.draw_known_annotations()
        # 如果存在交互点或掩膜，叠加回去
        if len(self.curr_inputs.input_point) != 0:
            self.display = self.du.draw_points(
                self.display, self.curr_inputs.input_point, self.curr_inputs.input_label
            )
        if self.curr_inputs.curr_mask is not None:
            self.display = self.du.overlay_mask_on_image(self.display, self.curr_inputs.curr_mask)

    def update_hover(self, x: int, y: int, margin: int = 6):
        """根据鼠标位置更新悬停的标注id，仅在接近时显示文字。
        规则：若点落在某个bbox的扩展区域（margin像素）内，选取距离中心最近的一个ann显示。
        返回值：是否发生变化（用于减少不必要的刷新）。
        """
        if not self.hover_mode_enabled:
            # 模式关闭时不更新悬停状态
            if len(self.hover_ann_ids) != 0:
                self.hover_ann_ids = set()
                self.display = self.image_bgr.copy()
                if self.show_other_anns:
                    self.draw_known_annotations()
                if len(self.curr_inputs.input_point) != 0:
                    self.display = self.du.draw_points(
                        self.display, self.curr_inputs.input_point, self.curr_inputs.input_label
                    )
                if self.curr_inputs.curr_mask is not None:
                    self.display = self.du.overlay_mask_on_image(self.display, self.curr_inputs.curr_mask)
                return True
            return False
        anns = self.dataset_explorer.get_annotations(self.image_id)
        if len(anns) == 0:
            changed = len(self.hover_ann_ids) != 0
            self.hover_ann_ids = set()
            return changed
        # 选择候选
        candidates = []
        for ann in anns:
            bx, by, bw, bh = ann["bbox"]
            bx, by, bw, bh = int(bx), int(by), int(bw), int(bh)
            if (bx - margin) <= x <= (bx + bw + margin) and (by - margin) <= y <= (by + bh + margin):
                cx, cy = bx + bw / 2.0, by + bh / 2.0
                dist2 = (x - cx) ** 2 + (y - cy) ** 2
                candidates.append((dist2, ann["id"]))
        new_hover = set()
        if candidates:
            candidates.sort(key=lambda t: t[0])
            new_hover.add(candidates[0][1])
        changed = new_hover != self.hover_ann_ids
        if not changed:
            return False
        self.hover_ann_ids = new_hover
        # 重绘显示（保留当前点击点和当前mask）
        self.display = self.image_bgr.copy()
        if self.show_other_anns:
            self.draw_known_annotations()
        # 重新叠加用户当前交互
        if len(self.curr_inputs.input_point) != 0:
            self.display = self.du.draw_points(
                self.display, self.curr_inputs.input_point, self.curr_inputs.input_label
            )
        if self.curr_inputs.curr_mask is not None:
            self.display = self.du.overlay_mask_on_image(self.display, self.curr_inputs.curr_mask)
        return True
    
    def step_up_transparency(self):
        self.du.increase_transparency()
        self.reset()

    def step_down_transparency(self):
        self.du.decrease_transparency()
        self.reset()

    def save_ann(self):
        self.dataset_explorer.add_annotation(
            self.image_id, self.category_id, self.curr_inputs.curr_mask
        )

    def delet_ann(self):
        self.dataset_explorer.delet_annotation(self.image_id)

    def save(self):
        self.dataset_explorer.save_annotation()

    def next_image(self):
        if self.image_id == self.dataset_explorer.get_num_images() - 1:
            return
        self.image_id += 1
        (
            self.image,
            self.image_bgr,
            self.image_embedding,
        ) = self.dataset_explorer.get_image_data(self.image_id)
        self.display = self.image_bgr.copy()
        self.reset()

    def prev_image(self):
        if self.image_id == 0:
            return
        self.image_id -= 1
        (
            self.image,
            self.image_bgr,
            self.image_embedding,
        ) = self.dataset_explorer.get_image_data(self.image_id)
        self.display = self.image_bgr.copy()
        self.reset()

    def go_to_image(self, idx: int):
        if self.dataset_explorer.get_num_images() == 0:
            return
        # 边界裁剪
        idx = max(0, min(idx, self.dataset_explorer.get_num_images() - 1))
        if idx == self.image_id:
            # 即使相同索引，也执行重置以确保显示一致
            self.reset()
            return
        self.image_id = idx
        (
            self.image,
            self.image_bgr,
            self.image_embedding,
        ) = self.dataset_explorer.get_image_data(self.image_id)
        self.display = self.image_bgr.copy()
        self.reset()

    def next_category(self):
        if self.category_id == len(self.categories) - 1:
            self.category_id = 0
            return
        self.category_id += 1

    def prev_category(self):
        if self.category_id == 0:
            self.category_id = len(self.categories) - 1
            return
        self.category_id -= 1
    
    def get_categories(self):
        return self.categories

    def select_category(self, category_name):
        category_id = self.categories.index(category_name)
        self.category_id = category_id
