import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class KuzushijiDataset(Dataset):
    """
    Dataset für CODH/Kuzushiji.
    Erwartet Struktur:
    
    data/
        images/<book_id>/*.jpg
        annotations/*.json
    """
    def __init__(self, root_dir, transform=None, label2id_path=None):
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir  # images liegen relativ zur annotations.json
        self.annotations_dir = self.root_dir / "annotations"
        # resize images to a manageable size for CPU smoke runs
        self.resize_to = (512, 512)
        self.transform = transform or T.Compose([T.ToTensor()])

        # Alle Annotationen sammeln
        self.ann_files = sorted(list(self.annotations_dir.glob("*.json")))
        self.img_paths = [self._get_img_path(f) for f in self.ann_files]

        # Labels laden oder Mapping erstellen
        if label2id_path:
            with open(label2id_path, "r", encoding="utf-8") as f:
                self.label2id = json.load(f)
        else:
            self.label2id = self._build_label_mapping()

        self.id2label = {v: k for k, v in self.label2id.items()}

    def _get_img_path(self, annot_file: Path):
        """
        Bestimmt den Bildpfad aus dem JSON-Dateinamen.
        z.B. 100241706_00004_2.json -> 100241706/images/100241706_00004_2.jpg
        """
        # Prefer explicit image_path in the annotation JSON when available
        try:
            with open(annot_file, "r", encoding="utf-8") as f:
                ann = json.load(f)
            img_path_rel = ann.get("image_path")
            if img_path_rel:
                return self.root_dir / img_path_rel
        except Exception:
            # fall back to filename-based resolution
            pass

        img_name = annot_file.name.replace(".json", ".jpg")
        # Extrahiere Book-ID aus Dateiname
        book_id = img_name.split("_")[0]
        return self.root_dir / book_id / "images" / img_name

    def _build_label_mapping(self):
        labels = set()
        for f in self.ann_files:
            with open(f, "r", encoding="utf-8") as jf:
                ann = json.load(jf)
            labels.update(ann.get("labels", []))
        return {l: i for i, l in enumerate(sorted(labels))}

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        annot_file = self.annotations_dir / (img_path.name.replace(".jpg", ".json"))

        # Bild laden
        image = Image.open(img_path).convert("RGB")

        # load annotations and rescale boxes if we resize the image
        with open(annot_file, "r", encoding="utf-8") as f:
            ann = json.load(f)
        orig_w, orig_h = image.size
        new_w, new_h = self.resize_to
        if (orig_w, orig_h) != (new_w, new_h):
            image = image.resize((new_w, new_h), resample=Image.BILINEAR)
            # scale boxes
            boxes = []
            for b in ann.get("boxes", []):
                x1, y1, x2, y2 = b
                sx = new_w / float(orig_w)
                sy = new_h / float(orig_h)
                boxes.append([x1 * sx, y1 * sy, x2 * sx, y2 * sy])
        else:
            boxes = ann.get("boxes", [])

        if self.transform:
            image = self.transform(image)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = [self.label2id[lbl] for lbl in ann.get("labels", [])]

        sample = {
            "image": image,
            "boxes": boxes,
            "labels": torch.tensor(labels, dtype=torch.long),
            "raw_labels": ann.get("labels", [])
        }
        return sample
