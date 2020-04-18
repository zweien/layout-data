Example
=======

采用 layout-generator 生成数据集，利用 FPN 模型进行 layout 到 heat_field
的回归

Usage
-----

-  查看参数帮助 ``python train.py -h``
-  使用配置文件 ``config.yml`` 进行训练。注意参数\ ``data_root``
   为数据集目录，需要包含子目录 ``train`` 与 ``test``

   .. code:: bash

      python train.py --config config.yml

-  训练过程中默认打印模型参数量，采用 ``tensorboard`` 记录
   ``hparams``\ ，\ ``train_loss``\ ，\ ``val_loss``\ ，model
   的输入与输出图像
-  ``tensorboard --logdir .`` 查看结果

Result
------

-  160 train，40 validation

-  参数

+---+---+------+---+---+---+-----+---+----+---+---+---+---+---+---+---+---+---+---+---+---+
| k | c | resu | s | g | u | val | t | da | t | d | i | m | s | m | s | m | o | l | b | n |
| e | o | me_f | e | p | s | _ch | e | ta | r | r | n | e | t | e | t | a | p | r | a | u |
| y | n | rom_ | e | u | e | eck | s | _r | a | o | p | a | d | a | d | x | t |   | t | m |
|   | f | chec | d | s | _ | _in | t | oo | i | p | u | n | _ | n | _ | _ | i |   | c | _ |
|   | i | kpoi |   |   | 1 | ter | _ | t  | n | _ | t | _ | l | _ | h | e | m |   | h | w |
|   | g | nt   |   |   | 6 | val | a |    | _ | p | _ | l | a | h | e | p | i |   | _ | o |
|   |   |      |   |   | b |     | r |    | s | r | s | a | y | e | a | o | z |   | s | r |
|   |   |      |   |   | i |     | g |    | i | o | i | y | o | a | t | c | e |   | i | k |
|   |   |      |   |   | t |     | s |    | z | b | z | o | u | t |   | h | r |   | z | e |
|   |   |      |   |   |   |     |   |    | e |   | e | u | t |   |   | s | _ |   | e | r |
|   |   |      |   |   |   |     |   |    |   |   |   | t |   |   |   |   | n |   |   | s |
|   |   |      |   |   |   |     |   |    |   |   |   |   |   |   |   |   | a |   |   |   |
|   |   |      |   |   |   |     |   |    |   |   |   |   |   |   |   |   | m |   |   |   |
|   |   |      |   |   |   |     |   |    |   |   |   |   |   |   |   |   | e |   |   |   |
+===+===+======+===+===+===+=====+===+====+===+===+===+===+===+===+===+===+===+===+===+===+
| v | c | None | 1 | 1 | F | 1   | F | d: | 0 | 0 | 2 | 5 | 1 | 2 | 2 | 2 | a | 0 | 6 | 4 |
| a | o |      |   |   | A |     | A | /w | . | . | 0 | 0 | 0 | 9 |   | 0 | d | . | 4 |   |
| l | n |      |   |   | L |     | L | or | 8 | 2 | 0 | 0 | 0 | 8 |   | 0 | a | 0 |   |   |
| u | f |      |   |   | S |     | S | k/ |   |   |   | 0 | 0 |   |   |   | m | 0 |   |   |
| e | i |      |   |   | E |     | E | da |   |   |   |   | 0 |   |   |   |   | 1 |   |   |
|   | g |      |   |   |   |     |   | ta |   |   |   |   |   |   |   |   |   |   |   |   |
|   | . |      |   |   |   |     |   | se |   |   |   |   |   |   |   |   |   |   |   |   |
|   | y |      |   |   |   |     |   | t1 |   |   |   |   |   |   |   |   |   |   |   |   |
|   | m |      |   |   |   |     |   |    |   |   |   |   |   |   |   |   |   |   |   |   |
|   | l |      |   |   |   |     |   |    |   |   |   |   |   |   |   |   |   |   |   |   |
+---+---+------+---+---+---+-----+---+----+---+---+---+---+---+---+---+---+---+---+---+---+

-  loss

|20200418125753| |20200418130347|

-  image

|20200418130456| |20200418130432| |20200418130527| |20200418130544|

.. |20200418125753| image:: https://i.loli.net/2020/04/18/8wpI7kWecGbslXU.png
.. |20200418130347| image:: https://i.loli.net/2020/04/18/LEd9aq7GNyPXUWm.png
.. |20200418130456| image:: https://i.loli.net/2020/04/18/xb6NIBY3HCfZ2lD.png
.. |20200418130432| image:: https://i.loli.net/2020/04/18/tAMjwzPT7YHDkVp.png
.. |20200418130527| image:: https://i.loli.net/2020/04/18/t9QcAh1ReLCkKSg.png
.. |20200418130544| image:: https://i.loli.net/2020/04/18/s4xvPZqIyDrNBAL.png

