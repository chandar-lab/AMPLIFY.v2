import torch
from torch.utils.data import DataLoader

from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset

from ..tokenizer import ProteinTokenizer

from .iterable_protein_dataset import IterableProteinDataset
from .data_collator import DataCollatorMLM


class CustomDataloader(DataLoader):
    """Dataloader with conversion of the pad mask from multiplicative to additive."""

    def __init__(self, dataset, dtype, **kwargs):
        super().__init__(dataset=dataset, **kwargs)
        self.dtype = dtype

    def __iter__(self):
        for batch in super().__iter__():
            batch["attention_mask"] = torch.where(
                batch["attention_mask"] == 1, torch.tensor(0.0, dtype=self.dtype), torch.tensor(float("-inf"), dtype=self.dtype)
            )
            yield batch


def get_dataloader(
    vocab_path: str,
    pad_token_id: int,
    mask_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    unk_token_id: int,
    other_special_token_ids: list | None,
    paths: dict,
    iterable: bool,
    pre_shuffle: bool,
    shuffle: bool,
    seed: int,
    on_the_fly_tokenization: bool,
    max_length: int,
    random_truncate: bool,
    return_position_ids: bool,
    remove_ambiguous: bool,
    num_workers: int,
    per_device_batch_size: int,
    mask_probability: int = 0,
    exclude_special_tokens_replacement: bool = True,
    pad_to_multiple_of: int = 8,
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> DataLoader:
    """Public wrapper for constructing a ``torch`` dataloader.

    Args:
        vocab_path (str): Path to the vocabulary file to load.
        pad_token_id (int): <PAD> token index in the vocab file.
        mask_token_id (int): <MASK> token index in the vocab file.
        bos_token_id (int): <BOS> token index in the vocab file.
        eos_token_id (int): <EOS> token index in the vocab file.
        unk_token_id (int): <UNK> token index in the vocab file.
        other_special_token_Unknown ids (list | None): List of other special tokens.
        paths (dict): Dict of name:paths to the CSV files to read.
        max_length (int): Maximum sequence length.
        random_truncate (bool): Truncate the sequence to a random subsequence of if longer than truncate.
        return_labels (bool): Return the protein labels.
        num_workers (int): Number of workers for the dataloader.
        per_device_batch_size (int): Batch size for each GPU.
        samples_before_next_set (list | None, optional): Number of samples of each dataset to return before moving
        to the next dataset (interleaving). Defaults to ``None``.
        mask_probability (int, optional): Ratio of tokens that are masked. Defaults to 0.
        span_probability (float, optional): Probability for the span length. Defaults to 0.0.
        span_max (int, optional): Maximum span length. Defaults to 0.
        exclude_special_tokens_replacement (bool, optional): Exclude the special tokens such as <BOS> or <EOS> from the
        replacement. Defaults to True.
        padding (str, optional): Pad the batch to the longest sequence or to max_length. Defaults to "max_length".
        pad_to_multiple_of (int, optional): Pad to a multiple of. Defaults to 8.
        dtype (torch.dtype, optional): Dtype of the pad_mask. Defaults to torch.float32.

    Returns:
        torch.utils.data.DataLoader
    """

    tokenizer = ProteinTokenizer(
        vocab_path,
        pad_token_id,
        mask_token_id,
        bos_token_id,
        eos_token_id,
        unk_token_id,
        other_special_token_ids,
    )
    collator = DataCollatorForLanguageModeling(
        tokenizer,
        mlm=True,
        mlm_probability=mask_probability,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors="pt",
    )

    dataset = load_dataset(
        "csv", data_files=list(paths.values()), keep_in_memory=False, num_proc=num_workers, split="all", streaming=iterable
    )

    def transform(inputs):
        return tokenizer.encode(
            inputs,
            max_length,
            random_truncate=random_truncate,
            return_position_ids=return_position_ids,
            remove_ambiguous=remove_ambiguous,
            return_special_tokens_mask=exclude_special_tokens_replacement,
        )

    if on_the_fly_tokenization:
        dataset.set_transform(transform)
    if pre_shuffle:
        dataset = dataset.shuffle(seed=seed)
        if not iterable:
            dataset = dataset.flatten_indices(num_proc=num_workers)

    return CustomDataloader(
        dataset=dataset,
        per_device_batch_size=per_device_batch_size,
        dtype=dtype,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=True,
    )
