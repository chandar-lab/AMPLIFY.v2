import torch
from typing import List, Optional, Union, Dict
from torch import Tensor

import numpy as np

# HuggingFace
from transformers import PreTrainedTokenizer


class ProteinTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab_path: str,
        pad_token_id: int,
        mask_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        unk_token_id: int,
        other_special_token_ids: Optional[List[int]],
        ambiguous_tokens: str = "XBOUZJ",
        **kwargs,
    ):
        """Vocabulary comprising the amino acids, and the special tokens <unk>, <bos>, <eos>, <pad> and <mask>.

        Args:
            vocab_path (str): Path to the vocabulary file to load.
            pad_token_id (int): <PAD> token index.
            mask_token_id (int): <MASK> token index.
            bos_token_id (int): <BOS> token index.
            eos_token_id (int): <EOS> token index.
            unk_token_id (int): <UNK> token index.
            other_special_token_ids (Optional[List[int]]): List of additional special tokens.
        """
        self._token_to_id = dict()
        self._id_to_token = dict()

        with open(vocab_path, "r") as vocab_file:
            for i, token in enumerate(vocab_file):
                token = token.strip()
                self._token_to_id[token] = i
                self._id_to_token[i] = token

        # Padding token
        self.pad_token_id = pad_token_id
        self.pad_token = self._token_to_id.get(pad_token_id)

        # Beginning and end of sequence
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.bos_token = self._token_to_id.get(bos_token_id)
        self.eos_token = self._token_to_id.get(eos_token_id)

        # Mask token
        self.mask_token_id = mask_token_id
        self.mask_token = self._token_to_id.get(mask_token_id)

        # Unknown token
        self.unk_token_id = unk_token_id
        self.unk_token = self._id_to_token.get(unk_token_id)

        # Set of all special token indices
        self.special_token_ids = set()
        self.special_token_ids.add(pad_token_id)
        self.special_token_ids.add(mask_token_id)
        self.special_token_ids.add(bos_token_id)
        self.special_token_ids.add(eos_token_id)
        self.special_token_ids.add(unk_token_id)
        if other_special_token_ids is not None:
            self.special_token_ids.update(other_special_token_ids)

        self.ambiguous_tokens_ids = [self._token_to_id[tok] for tok in ambiguous_tokens]
        self.ambiguous_tokens_ids.append(self.unk_token_id)

    def __len__(self) -> int:
        return len(self._token_to_id)

    def token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self.unk_token_id)

    def id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    def encode(
        self,
        tokens: List[str],
        max_length: Optional[int] = None,
        add_special_tokens: bool = True,
        random_truncate: bool = True,
        return_position_ids: bool = False,
        remove_ambiguous: bool = False,
        return_special_tokens_mask: bool = True,
        **kwargs,
    ) -> Dict[Tensor]:
        """Encodes a list of tokens into a list or tensor of token indices.

        Args:
            tokens (List[str]): Sequence of tokens to encode.
            max_length (Optional[int], optional): Truncate the sequence to the specified length. Defaults to None.
            add_special_tokens (bool, optional): Add special tokens <bos> and <eos> at the start and end. Defaults to True.
            random_truncate (bool, optional): Truncate the sequence to a random subsequence if longer than max_length.
            remove_ambiguous (bool, optional):
            return_special_tokens_mask (bool, optional):
            Defaults to True.

        Returns:
            Union[List[int], Tensor]: Token indices.
        """
        # Convert to ids
        token_ids = list(map(self.token_to_id, tokens))

        # Add BOS and EOS tokens
        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]

        # Truncate
        offset = 0
        if max_length is not None and max_length < len(token_ids):
            if random_truncate:
                offset = int(torch.randint(0, len(token_ids) - max_length, (1,)).item())
            token_ids = token_ids[offset : offset + max_length]

        token_ids = torch.as_tensor(token_ids, dtype=torch.long)

        # Store the position indexes
        if return_position_ids:
            position_ids = torch.as_tensor(range(offset, offset + max_length), dtype=torch.long)

        # Optionally remove ambiguous or unknown amino acids
        if remove_ambiguous:
            mask = ~np.isin(token_ids, self.ambiguous_tokens_ids)

            token_ids = token_ids[mask]
            if return_position_ids:
                position_ids = position_ids[mask]

        output = {"input_ids": token_ids}
        if return_position_ids:
            output["input_positions"] = position_ids
        if return_special_tokens_mask:
            output["special_tokens_mask"] = torch.isin(token_ids, torch.tensor(self.special_token_ids))

        return output

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> Union[List[str], str]:
        """Decodes a list or tensor of token ids into a list or string of tokens.

        Args:
            token_ids (List[int]): Token indices to decode.
            skip_special_tokens (bool, optional): Skip the special tokens <bos> and <eos> at the start and end.
            Defaults to True.

        Returns:
            Union[List[str], str]: Protein.
        """
        if torch.is_tensor(token_ids):
            token_ids = token_ids.tolist()

        if skip_special_tokens:
            if len(token_ids) > 0 and token_ids[0] in self.special_token_ids:
                token_ids = token_ids[1:]
            if len(token_ids) > 0 and token_ids[-1] in self.special_token_ids:
                token_ids = token_ids[:-1]

        tokens = " ".join(map(self.id_to_token, token_ids))

        return tokens
