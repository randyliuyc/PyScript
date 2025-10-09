# TotalLINK Python 调用功能
# 统一用 linkrun 函数执行功能调用
# 调用参数为 JSON 格式的字符串，例如 '{"A": 3, "B": 4, "Value": [1, 2, 3, 4]}'

def linkrun(json_str):
    """
    处理传入的 JSON 字符串，并输出相关信息
    :param json_str: JSON 格式的字符串，例如 '{"A": 3, "B": 4, "Value": [1, 2, 3, 4]}'
    :return: 返回处理后的信息字符串
    """
    import json
    try:
        data = json.loads(json_str)

        result = {
            "isSuccess": True,
            "message": f"收到参数: {json_str}"
        }
        return json.dumps(result)  # 返回 JSON 字符串

    except json.JSONDecodeError as e:
        result = {
            "isSuccess": False,
            "message": f"JSON 解析错误: {e}"
        }
        return json.dumps(result)  # 返回 JSON 字符串
    
if __name__ == "__main__":
    print("Starting ...")

    json_str = '{"A": 3, "B": 4, "Value": [1, 2, 3, 4],}'
    result = linkrun(json_str)
    print(result)    