@echo off
REM Headless Gemma 12B launcher — no menu, fixed "none" reasoning profile.
REM The VN app relaunches the server with these exact args during the
REM cast-generation VRAM swap. You can also run this by hand.
REM Override the model/exe by editing the two paths below.

set MODEL=C:\Users\vruizes\Downloads\gemma-4-12b-it-IQ4_XS.gguf
set LLAMA=C:\Users\vruizes\llama-cpp-setup\llama-server.exe

"%LLAMA%" -m "%MODEL%" -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 16384 --reasoning off -np 1 --cache-ram 0 --host 0.0.0.0 --port 8111
