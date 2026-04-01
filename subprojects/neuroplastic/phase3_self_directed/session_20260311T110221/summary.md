# Session Summary: session_20260311T110221

**Turns:** 42
**Modifications:** 17
**Evaluations:** 15

## Evaluations
- 2026-03-11T11:11:10.115972: 83.3% (10/12) mode=quick
- 2026-03-11T11:15:31.274928: 75.0% (9/12) mode=quick
- 2026-03-11T11:19:57.894016: 83.3% (10/12) mode=quick
- 2026-03-11T11:24:56.679518: 83.3% (10/12) mode=quick
- 2026-03-11T11:31:14.406067: 66.7% (8/12) mode=quick
- 2026-03-11T11:37:54.813648: 83.3% (10/12) mode=quick
- 2026-03-11T11:43:33.483295: 75.0% (9/12) mode=quick
- 2026-03-11T11:49:03.966562: 91.7% (11/12) mode=quick
- 2026-03-11T11:56:08.676679: 91.7% (11/12) mode=quick
- 2026-03-11T12:00:22.942793: 75.0% (9/12) mode=quick
- 2026-03-11T12:16:39.840135: 75.0% (9/12) mode=full
- 2026-03-11T12:31:44.235356: 83.3% (10/12) mode=full
- 2026-03-11T12:46:15.080875: 83.3% (10/12) mode=full
- 2026-03-11T13:01:39.207924: 83.3% (10/12) mode=full
- 2026-03-11T13:12:47.967800: 75.0% (9/12) mode=quick

## Modifications
- MODIFY model.layers.46.mixer.A op=scale_slice value=? → ok
- MODIFY model.layers.46.mixer.A op=scale_slice value=? → ok
- MODIFY model.layers.46.mixer.A op=scale_slice value=? → ok
- MODIFY model.layers.48.mixer.A op=scale_slice value=? → ok
- MODIFY model.layers.48.mixer.A op=scale_slice value=? → ok
- MODIFY model.layers.48.mixer.D op=scale value=? → ok
- MODIFY model.layers.48.mixer.D op=scale value=? → ok
- MODIFY model.layers.50.mixer.A op=scale_slice value=? → ok
- MODIFY model.layers.50.mixer.D op=scale value=? → ok
- MODIFY model.layers.50.mixer.A op=scale_slice value=? → ok
- MODIFY model.layers.50.mixer.D op=scale value=? → ok
- MODIFY model.layers.51.mixer.gate.weight op=scale_rows value=? → ok
- MODIFY model.layers.42.mixer.A op=scale_slice value=? → Tensor 'model.layers.42.mixer.A' not found
- MODIFY model.layers.50.mixer.D op=scale value=? → ok
- MODIFY model.layers.46.mixer.A op=scale value=? → ok
- MODIFY model.layers.50.mixer.A op=scale_slice value=? → ok
- MODIFY model.layers.50.mixer.A op=zero_heads value=? → ok
