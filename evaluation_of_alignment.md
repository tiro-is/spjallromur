# Realignment of Spjallromur

The dataset was realgined using scripts in Kaldi and an acoustic model from Tiro ehf.

# Evaluation of alignments

The data was evaluated in two ways. Firstly by decoding using an ASR system and then also by manually listning to a few hundred segments.

## Evaluation by decoding

We used a finetuned Whisper model (https://huggingface.co/language-and-voice-lab/whisper-large-icelandic-30k-steps-1000h).

The average word error rate of the entire dataset is 38.63% which is quite high and results for each speaker are in the table below. This model performs well on other test sets and this is therefore a quite a high WER. There are some noticeble outliers in data. For example the speaker `b_9a67bc98_30-39_f` has Icelandic as their second language and therefore has a thick accent.

| Spk id             | WER     |
| ------------------ | ------- |
| b_9a67bc98_30-39_f | 163.713 |
| b_24c0c1b3_20-29_m | 70.577  |
| a_ebbc5293_20-29_m | 69.91   |
| a_2c1b4416_40-49_f | 65.217  |
| b_2a07b3a7_20-29_f | 62.906  |
| b_5f55950e_20-29_f | 62.162  |
| a_92a95e84_30-39_m | 61.454  |
| a_eda6925c_30-39_m | 60.44   |
| a_2ecf1db5_20-29_m | 58.198  |
| a_1939f519_20-29_m | 58.067  |
| b_92a95e84_30-39_m | 57.94   |
| a_b6ad9f96_20-29_f | 56.503  |
| b_dc0967ee_30-39_m | 54.291  |
| b_eda6925c_30-39_m | 52.725  |
| a_2f1655ff_40-49_f | 52.416  |
| a_01119679_40-49_f | 52.263  |
| b_81b2b35e_30-39_m | 49.907  |
| a_2a07b3a7_20-29_m | 49.237  |
| b_ebbc5293_20-29_m | 49.057  |
| a_0f2c315c_30-39_m | 47.826  |
| a_66ccf3bc_20-29_m | 45.914  |
| a_ad46e29b_20-29_m | 45.81   |
| a_05b30647_30-39_m | 43.48   |
| a_c3a7fbe9_20-29_m | 43.014  |
| b_aad7caab_30-39_m | 42.371  |
| b_1939f519_20-29_m | 41.837  |
| a_69079ee1_30-39_m | 41.693  |
| a_7faf84e8_30-39_m | 40.616  |
| a_44d73360_30-39_m | 40.37   |
| b_05b30647_30-39_f | 40.35   |
| a_de3b604f_20-29_m | 39.735  |
| b_a56ed5af_60-69_f | 39.606  |
| b_5331448b_30-39_m | 39.585  |
| b_188092d3_20-29_m | 39.216  |
| a_aad7caab_20-29_m | 39.211  |
| b_ccd0f1a6_30-39_f | 39.152  |
| b_deb42548_20-29_f | 38.751  |
| a_e1e7765a_20-29_f | 38.606  |
| b_de3b604f_20-29_m | 38.322  |
| a_879325a8_20-29_m | 37.622  |
| b_2ecf1db5_20-29_o | 37.089  |
| b_826b4d3d_40-49_m | 37.069  |
| b_2f1655ff_40-49_f | 36.728  |
| a_8edc23bf_20-29_m | 36.504  |
| b_8af8f246_20-29_m | 36.266  |
| b_2c1b4416_30-39_f | 36.173  |
| a_389f0bb5_20-29_f | 36.082  |
| a_b107d272_30-39_m | 35.953  |
| b_b107d272_20-29_m | 35.697  |
| b_44d73360_30-39_m | 35.394  |
| a_dc0967ee_30-39_m | 35.342  |
| a_ccd0f1a6_30-39_m | 35.044  |
| b_81dd3246_30-39_f | 35.007  |
| b_8edc23bf_30-39_m | 34.806  |
| a_8c25247b_20-29_m | 34.697  |
| b_bcb44230_30-39_m | 34.472  |
| b_2d219d50_30-39_m | 34.32   |
| a_a56ed5af_20-29_f | 34.271  |
| b_c3a7fbe9_20-29_m | 33.028  |
| b_e25bc38d_20-29_m | 32.798  |
| a_2d219d50_30-39_m | 32.659  |
| a_188092d3_30-39_f | 32.47   |
| a_2a139f9b_20-29_m | 32.451  |
| b_8c25247b_30-39_m | 32.444  |
| b_01119679_50-59_m | 32.237  |
| a_826b4d3d_30-39_m | 31.758  |
| b_69079ee1_30-39_m | 31.299  |
| a_deb42548_20-29_m | 31.108  |
| a_bbc3f248_20-29_m | 31.073  |
| b_879325a8_20-29_m | 31.041  |
| a_5f55950e_20-29_f | 30.853  |
| a_54ddefa8_30-39_f | 30.416  |
| a_3ac74ae1_40-49_f | 30.063  |
| b_389f0bb5_20-29_f | 29.851  |
| b_66ccf3bc_50-59_m | 29.564  |
| a_2284fd64_40-49_f | 29.095  |
| b_0f2c315c_30-39_f | 28.894  |
| b_54ddefa8_30-39_m | 28.075  |
| b_3ce6563e_20-29_m | 27.937  |
| a_f123a375_20-29_f | 27.273  |
| a_caa6301e_20-29_m | 26.718  |
| a_24c0c1b3_20-29_f | 26.46   |
| b_2284fd64_40-49_m | 25.86   |
| a_5331448b_30-39_m | 25.76   |
| a_50d1de3c_20-29_o | 25.136  |
| a_45eebf55_30-39_f | 24.843  |
| b_7faf84e8_30-39_f | 24.684  |
| a_81b2b35e_30-39_m | 24.47   |
| a_bcb44230_20-29_m | 24.108  |
| b_45eebf55_50-59_m | 24.051  |
| b_ad46e29b_20-29_m | 23.427  |
| a_997d4fe0_30-39_m | 23.231  |
| b_50d1de3c_20-29_f | 22.377  |
| a_3ce6563e_50-59_m | 22.217  |
| a_8af8f246_20-29_m | 22.178  |
| a_81dd3246_30-39_f | 20.583  |
| b_997d4fe0_30-39_m | 18.182  |
| b_bbc3f248_20-29_m | 17.854  |
| a_9a67bc98_80-89_m | 15.834  |

## Evaluation by manual listening

Five samples were taken from each speaker and manually evaluated. The samples were chosen randomly from each speaker. But due time constraints not all speakers were tested. In total there we evaluated 296 samples from 60 speakers ot ouf a 99. The evaluation was very simple, either there was a good alignment, the beginning of the segment was missing or the end of the segment was missing. The results are quite good and the average accuracy is 94.6% most of the alignment issues were minor.

| Evaluation        | Count |
| ----------------- | ----- |
| Good              | 296   |
| Missing end       | 12    |
| Missing beginning | 4     |

## Conclusion

The manual evaluation gives us confidence in the alignment. The WER is quite high but the model is not trained on this data and the data is quite different from the data the model was trained on. This conversational data differs a lot from traditional ASR data, as the speakers are in some cases not speaking clearly, the audio quality is varied, the speakers are not reading from a script and the sentence structure is very unorthodox. As can be seen in these samples.

```
a_2c1b4416_40-49_f_0_18.86
<UNK>. Mér finnst þessi rauði þarna, þetta er dálítið villandi, við hliðina á. Þess vegna erum við ekki að senda þetta út á trilljón manns, við erum enn þá laga einhverja bögga.

b_aad7caab_30-39_m_64_19.64
Já, já, já, já, nákvæmlega. Og þurfti að fara, ég man eftir því að þurfti að þú veist actually fara út og tékka á miklu meira dóti sko. Eitthvað svona lítil mission bara eitthvað. Þetta er svona. Ég veit það ekki.
```
