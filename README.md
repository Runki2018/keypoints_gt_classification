# keypoints_gt_classification
用关键点真值直接训练一个简单的全连接网络，用于测试用关键点做手势分类的可行性。



> **1、confidence_threshold = 0.95**
>
>  <table>
>  <tr>
>   <td align="center"><img src="./doc_img/1.png" width="100%" height="auto" /></td>
>  </tr>
> </table>
>
>  <table>
>  <tr>
>   <td align="center"><img src="./doc_img/2.png" width="100%" height="auto" /></td>
>  </tr>
> </table>
>
> 2、confidence_threshold = 0.98
>
>  <table>
>  <tr>
>   <td align="center"><img src="./doc_img/3.png" width="60%" height="auto" /></td>
>  </tr>
> </table>
>
>  <table>
>  <tr>
>   <td align="center"><img src="./doc_img/4.png" width="100%" height="auto" /></td>
>  </tr>
> </table>

------

最终决定用0.99的置信度阈值来筛选错分样本，在所有的测试样本中还有18个错分样本。

对错分样本进行分析，基本上就只剩下“0-其他”类和其余八类之间的错分，然后观察这18张图片的特点属于比较容易混淆的类别，以当前分类的准确率AP=0.9897来看，用关键点来辅助手势分类是可行的。

