"""Vocabulary manager with special tokens for sequence modeling."""
import json
from pathlib import Path
from typing import List, Dict, Optional


class VocabManager:
    """Manages vocabulary with special tokens (PAD, SOS, EOS) for sequence tasks."""
    
    PAD_TOKEN = '<PAD>'
    SOS_TOKEN = '<SOS>'
    EOS_TOKEN = '<EOS>'
    UNK_TOKEN = '<UNK>'
    BG_TOKEN  = '<BG>'     # background: FP proposals trained to predict this
    
    def __init__(self, char2id: Optional[Dict[str, int]] = None):
        """Initialize vocab. If char2id not provided, starts with just special tokens."""
        if char2id is None:
            # Start with special tokens
            self.char2id = {
                self.PAD_TOKEN: 0,
                self.SOS_TOKEN: 1,
                self.EOS_TOKEN: 2,
                self.UNK_TOKEN: 3,
            }
        else:
            self.char2id = char2id
        
        self.id2char = {v: k for k, v in self.char2id.items()}
        
    @property
    def pad_id(self) -> int:
        return self.char2id[self.PAD_TOKEN]
    
    @property
    def sos_id(self) -> int:
        return self.char2id[self.SOS_TOKEN]
    
    @property
    def eos_id(self) -> int:
        return self.char2id[self.EOS_TOKEN]
    
    @property
    def unk_id(self) -> int:
        return self.char2id[self.UNK_TOKEN]

    @property
    def bg_id(self) -> int:
        return self.char2id[self.BG_TOKEN]
    
    @property
    def vocab_size(self) -> int:
        return len(self.char2id)
    
    def add_char(self, char: str) -> int:
        """Add a character to vocab if not present. Returns its ID."""
        if char not in self.char2id:
            self.char2id[char] = len(self.char2id)
            self.id2char[self.char2id[char]] = char
        return self.char2id[char]
    
    def encode(self, text: List[str], add_sos: bool = True, add_eos: bool = True) -> List[int]:
        """Encode list of characters to IDs with optional SOS/EOS tokens."""
        ids = []
        if add_sos:
            ids.append(self.sos_id)
        for char in text:
            ids.append(self.char2id.get(char, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids
    
    def decode(self, ids: List[int], remove_special: bool = True) -> List[str]:
        """Decode IDs to characters, optionally removing special tokens."""
        special_ids = {self.pad_id, self.sos_id, self.eos_id}
        if remove_special and self.BG_TOKEN in self.char2id:
            special_ids.add(self.bg_id)
        chars = []
        for id_ in ids:
            if remove_special and id_ in special_ids:
                continue
            chars.append(self.id2char.get(id_, self.UNK_TOKEN))
        return chars
    
    @classmethod
    def from_annotations(cls, annotation_files: List[Path]) -> 'VocabManager':
        """Build vocabulary from annotation files containing 'labels' field."""
        vocab = cls()
        unique_chars = set()
        
        for ann_file in annotation_files:
            with open(ann_file, 'r', encoding='utf-8') as f:
                ann = json.load(f)
            labels = ann.get('labels', [])
            unique_chars.update(labels)
        
        # Add all characters in sorted order for reproducibility
        for char in sorted(unique_chars):
            vocab.add_char(char)

        # BG token is always appended last so regular char IDs are stable
        vocab.add_char(cls.BG_TOKEN)

        return vocab
    
    def save(self, path: Path):
        """Save vocabulary to JSON."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.char2id, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'VocabManager':
        """Load vocabulary from JSON."""
        with open(path, 'r', encoding='utf-8') as f:
            char2id = json.load(f)
        return cls(char2id)
