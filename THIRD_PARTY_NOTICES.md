# Third-party notices

This repository contains no third-party Swift code and has no Swift package dependencies.

## Real-ESRGAN

The model architectures in `tools/convert/model_loader.py` (RRDBNet and SRVGGNetCompact) are
reimplemented from the public description in the Real-ESRGAN project, with module and parameter
names that follow upstream so its published checkpoints load with `strict=True`. No code from
Upscayl (AGPL-3.0) or any other GPL/AGPL source was consulted or copied.

- Project: <https://github.com/xinntao/Real-ESRGAN>
- Licence: BSD 3-Clause — reproduced below and in `models/RealESRGAN_x4plus/LICENSE`

```
BSD 3-Clause License

Copyright (c) 2021, Xintao Wang
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

This project is not endorsed by, or affiliated with, the Real-ESRGAN authors.

## Model weights

**No model weights are included in this repository.** `tools/artifacts/fetch.py` downloads them
from the upstream project's GitHub releases, at the URLs and SHA256 hashes pinned in
`models/<id>/manifest.json`, and the converted `.mlpackage` files are generated locally.

The terms that apply to pre-trained weights are set by their publishers and by the datasets they
were trained on, and they are not necessarily the same as the code licence above. Check the
upstream project's licence and its training-data terms before redistributing weights or converted
models, and before any commercial use.

## Demo images

The images in `docs/images/` are adaptations (downscaled, upscaled and cropped) of two photographs
by Yasir Dora, published on Wikimedia Commons under CC BY-SA 4.0
(<https://creativecommons.org/licenses/by-sa/4.0/>), and are shared under the same licence:

- "Sindh monument at Chaukhandi, Pakistan" —
  <https://commons.wikimedia.org/wiki/File:Sindh_monument_at_chaukhandi_pakistan.jpg>
- "Date palm tree, Balochistan, Pakistan" —
  <https://commons.wikimedia.org/wiki/File:Date_palm_tree_balochistan_pakistan.jpg>

## Development tooling

The Python tools under `tools/` run on a development machine only; nothing in them is linked into
the Swift packages. Their pinned dependencies (`tools/convert/requirements.txt`) are distributed
under their own licences — PyTorch, torchvision, coremltools, NumPy, SciPy, scikit-image, imageio,
tifffile and lazy-loader (BSD family), Pillow (MIT-CMU), PyYAML and pytest (MIT), and `lpips`
(BSD-2-Clause, whose AlexNet trunk weights are downloaded on first use and used for evaluation
only).
