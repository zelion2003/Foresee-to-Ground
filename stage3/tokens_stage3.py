#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import List, Tuple


def add_span_tokens(tokenizer, model, k: int = 8, add_span_pad: bool = True) -> Tuple[List[int], int]:
    
    new_tokens = [f"<Span_{i}>" for i in range(k)]
    if add_span_pad:
        new_tokens.append("<span_pad>")
    old_vocab = model.get_input_embeddings().weight.size(0)
    tokenizer.add_tokens(new_tokens, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    emb = model.get_input_embeddings().weight

    def zero_old_rows_grad(grad):
        grad[:old_vocab].zero_()
        return grad

    emb.register_hook(zero_old_rows_grad)
    new_ids = [tokenizer.convert_tokens_to_ids(t) for t in new_tokens]
    return new_ids, old_vocab
