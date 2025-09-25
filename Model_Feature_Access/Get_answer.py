"""
需求文档：基于图像和问题的自动化问答与准确率评估

1. 数据读取
   - 读取问题文件：路径为 /Users/yize/Documents/chenyize/python_project/X2DFD/question.json
     - 文件内容为问题列表，每个问题为字符串，例如：
       [
         "Are the eyes unnaturally shaped or asymmetrical? Answer Yes or No.",
         ...
       ]
   - 读取真实图片信息文件：路径为 /Users/yize/Documents/chenyize/python_project/X2DFD/real.json
     - 文件内容为包含图片路径的字典，例如：
       {
         "Description": "",
         "images": [
           {"image_path": "Celeb-DF-v3-process/Celeb-real/id0_0000/000.png"},
           ...
         ]
       }
   - 读取伪造图片信息文件：路径为 /Users/yize/Documents/chenyize/python_project/X2DFD/fake.json
     - 文件结构与 real.json 相同。

2. 问答模拟
   - 实现一个模拟问答函数 gen_answer(image_path, question)
     - 输入：图片路径和问题
     - 输出：随机返回 "Yes" 或 "No"

3. 准确率计算
   - 对于 fake.json 中的所有图片和问题，统计 gen_answer 返回 "Yes" 的次数，记为 Yes_count
   - 计算伪造图片的准确率：Fake_ACC = Yes_count / Fake_num
     - Fake_num 为伪造图片总数
   - 对真实图片（real.json）同样计算准确率

4. 输出
   - 输出real acc 和fake acc的balance acc score

"""