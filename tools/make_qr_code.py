import os

import qrcode
import numpy as np
rng = np.random.default_rng()


class ToolMakeQRCode():
    def __init__(self):
        self.name = 'make_qr_code'

        self.tool_description = """
<tool_description>
<tool_name>{{TOOLNAME}}</tool_name>
<description>Generates an image of a QR code given the text to be coded and the QR code configurations.

The user can view the QR code generated without exposing the auto-generated file names. Do NOT include actual file names in the answer. Do NOT include the <path_to_image></path_to_image> tag in the answer. Only mention that the QR code has been generated successfully.

Raises ValueError: if any parameter was invalid.
</description>

<parameters>
<parameter>
<name>qr_text</name>
<type>string</type>
<description>String to be encoded in the QR code image
</description>
</parameter>

<parameter>
<name>error_correction</name>
<type>string</type>
<description>Optional. If specified, must be one of the <correction_values>. Lower values generate smaller QR codes but are less tolerant to faults. Higher values generate larger QR codes but the QR code is still valid in the presence of more damage to the image. If no error_correction value is provided, the default 'medium' will be used.

<correction_values>
<value>
<name>low</name>
<description>About 7% or less errors can be corrected (low).</description>
</value>
<value>
<name>medium</name>
<description>Default. About 15% or less errors can be corrected (medium).</description>
</value>
<value>
<name>high</name>
<description>About 25% or less errors can be corrected (high).</description>
</value>
<value>
<name>highest</name>
<description>About 30% or less errors can be corrected (highest).</description>
</value>
</correction_values>
</description>
</parameter>

<parameter>
<name>box_size</name>
<type>integer</type>
<description>Optional. If specified, controls how many pixels each "box" of the QR code has. The size of the image increases for larger values.
The default value is 10.
Only use values from 5 (small image) to 50 (very large image).
</description>
</parameters>

</tool_description>
        """.replace('{{TOOLNAME}}', self.name)

    def __call__(self, qr_text, error_correction='medium', box_size=10, **kwargs):
        if len(kwargs) > 0:
            return f"Error: Unexpected parameter(s): {','.join([x for x in kwargs])}"

        error_corrections = {
            'low': qrcode.constants.ERROR_CORRECT_L,
            'medium': qrcode.constants.ERROR_CORRECT_M,
            'high': qrcode.constants.ERROR_CORRECT_Q,
            'highest': qrcode.constants.ERROR_CORRECT_H,
        }
        if error_correction not in error_corrections.keys():
            return f'error_correction must be one of {error_corrections.keys()}'

        qr = qrcode.QRCode(
            version=1,
            error_correction=error_corrections[error_correction],
            box_size=10,
            border=4,
        )
        qr.add_data(qr_text)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")

        rng_num = rng.integers(low=0, high=900000)
        target_file = f"media/qr_{rng_num}.png"
        img.save(target_file)

        if not os.path.isfile(target_file):
            return "Error: Image was not saved correctly."

        #with open(f'plot_code_{rng_num}.txt', 'w', encoding='utf-8') as debug_file:
        #    p_str = plot_code + f'\n***{target_file} {os.path.isfile(target_file)}'
        #    debug_file.write(p_str)
        
        ans = ["<image>"]
        ans.append(f"<path_to_image>{target_file}</path_to_image>")
        ans.append("</image>")
        return '\n'.join(ans)
