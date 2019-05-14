# Chinese License Plate recognition

This is Intel OpenVINO demo to demonstrate Chinese LPR.

Requirements:
* Intel&reg; Neural Compute Stick 2
* (optional) LCD with I2C adapter

To control the LCD is used library by Denis Pleic: https://gist.github.com/DenisFromHR/cc863375a6e19dce359d

In my experiments with Raspberry Pi 3 I just enabled I2C interface in `raspi-config` (`5. Interfacing Options` -> `I2C`)
and used the following pins connections: SDA (pin 2), SCL (pin 3). See [specifications](https://www.raspberrypi.org/documentation/usage/gpio/):

```bash
$ i2cdetect -y 1
     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
00:          -- -- -- -- -- -- -- -- -- -- -- -- --
10: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
20: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
30: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 3f
40: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
50: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
60: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
70: -- -- -- -- -- -- -- --
```

If your output is different, change LCD address in `RPi_I2C_driver.py` correspondingly:

```python
# LCD Address
ADDRESS = 0x3f
```
