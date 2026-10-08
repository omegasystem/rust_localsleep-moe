import torch

def trim_past_key_values_swa(past_key_values, sink_size: int = 4, window_size: int = 2048):
    """
    Sink-Aware Sliding Window Attention (SWA) In-Place Bounded KV-Cache Slicing.
    Preserves: [..., :sink_size, :] (Sinks) + [..., -window_size:, :] (Local Window).
    Supports:
      1. Transformers 4.45+ DynamicCache (.layers list of CacheLayer / DynamicCache.key_cache)
      2. Legacy DynamicCache (_key_cache / key_cache attribute)
      3. Legacy tuple-of-tuples KV cache
    """
    if past_key_values is None:
        return None

    max_len = sink_size + window_size

    # Case 1: Transformers modern DynamicCache with .layers (Thor / Transformers 4.45+)
    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            if hasattr(layer, "keys") and layer.keys is not None:
                cur_len = layer.keys.shape[-2]
                if cur_len > max_len:
                    k_sink = layer.keys[..., :sink_size, :]
                    k_win = layer.keys[..., -window_size:, :]
                    layer.keys = torch.cat([k_sink, k_win], dim=-2)

                    v_sink = layer.values[..., :sink_size, :]
                    v_win = layer.values[..., -window_size:, :]
                    layer.values = torch.cat([v_sink, v_win], dim=-2)
        return past_key_values

    # Case 2: Intermediate DynamicCache with key_cache / _key_cache attribute
    key_cache = getattr(past_key_values, "key_cache", None) or getattr(past_key_values, "_key_cache", None)
    value_cache = getattr(past_key_values, "value_cache", None) or getattr(past_key_values, "_value_cache", None)

    if key_cache is not None and value_cache is not None:
        for idx in range(len(key_cache)):
            k = key_cache[idx]
            v = value_cache[idx]
            cur_len = k.shape[-2]
            if cur_len > max_len:
                k_sink = k[..., :sink_size, :]
                k_win = k[..., -window_size:, :]
                key_cache[idx] = torch.cat([k_sink, k_win], dim=-2)

                v_sink = v[..., :sink_size, :]
                v_win = v[..., -window_size:, :]
                value_cache[idx] = torch.cat([v_sink, v_win], dim=-2)
        return past_key_values

    # Case 3: Legacy tuple / list of (key, value) pairs
    if isinstance(past_key_values, (tuple, list)):
        new_pkv = []
        for layer in past_key_values:
            if isinstance(layer, (tuple, list)) and len(layer) >= 2:
                k, v = layer[0], layer[1]
                cur_len = k.shape[-2]
                if cur_len > max_len:
                    k = torch.cat([k[..., :sink_size, :], k[..., -window_size:, :]], dim=-2)
                    v = torch.cat([v[..., :sink_size, :], v[..., -window_size:, :]], dim=-2)
                new_pkv.append((k, v))
            else:
                new_pkv.append(layer)
        return tuple(new_pkv) if isinstance(past_key_values, tuple) else new_pkv

    return past_key_values
