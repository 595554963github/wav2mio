# wav2mio
Entis MIO encoder

miocompress.exe and mioplayer.exe come from https://www.rarewares.org/rrw/mio.php. wav2mio.exe comes from https://github.com/julixian/MyVisualNovelTransTools.

According to the documentation at https://github.com/lxl66566/lxl66566.github.io/blob/6b4dc72baa4ede1739c5f5a31dff0eab4fbf5840/src/articles/speedup.md?plain=1#L1101, it originates from the EntisGLS engine. 

However, to make things easier for other developers, I reverse-engineered wav2mio.exe and produced this wav2mio.cpp. 

The official wav2mio.exe is 1,085,440 bytes — exactly 1 MB. My version, compiled in Release mode, is only 37,888 bytes, or 37.0 KB — a 28.6× reduction in size.

If you want to call my unofficial version in your own project, it can greatly reduce your project's size. 

In addition, you can port wav2mio.cpp to C# or other languages. All three MIO encoders are valid — the choice is yours.

Good luck!
