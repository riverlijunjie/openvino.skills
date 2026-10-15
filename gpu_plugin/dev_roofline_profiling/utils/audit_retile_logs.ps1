$tags = @(
 "conv_dec_conv_in","conv_dec_mid_conv1","conv_dec_up0_conv1","conv_dec_up0_upscale_conv",
 "conv_dec_up0_post_shuffle_conv","conv_dec_up1_conv1","conv_dec_up1_upscale_conv",
 "conv_dec_up1_post_shuffle_conv","conv_dec_up2_shortcut","conv_dec_up2_conv1","conv_dec_up2_conv2",
 "conv_dec_up2_post_shuffle_conv","conv_dec_up3_shortcut","conv_dec_up3_conv1","conv_dec_up3_conv2",
 "conv_dec_conv_out","gn_dec_gn_c512_h72w122","gn_dec_gn_c512_h144w244","gn_dec_gn_c256_h288w488",
 "gn_dec_gn_c128_h576w976","fc_dec_mid_attn_qkvo","attn_dec_mid_attn_sdpa"
)
foreach ($t in $tags) {
  $p = "D:\river\ovmx_dev\roofline\results\seedvr2\$t.log"
  if (Test-Path $p) {
    $f = Get-Item $p
    Write-Output ("{0,-35} {1} {2,10} bytes" -f $t, $f.LastWriteTime, $f.Length)
  } else {
    Write-Output ("{0,-35} MISSING" -f $t)
  }
}
