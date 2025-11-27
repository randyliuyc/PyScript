import requests
import json
import pandas as pd
from io import StringIO

# Superset 连接信息
SUPERSET_URL = "http://124.71.144.80:9047"
s = requests.Session()

# 登录
login_payload = {"username": "randy", "password": "randy", "provider": "db"}
login_response = s.post(f"{SUPERSET_URL}/api/v1/security/login", json=login_payload)
print(f"✓ 登录成功")

def create_json_dataset_from_api(api_url, dataset_name):
    """从 JSON API 创建数据集"""
    
    # 1. 获取 JSON 数据
    print(f"\n=== 从 API 获取数据: {api_url} ===")
    try:
        api_response = requests.get(api_url, timeout=10)
        if api_response.status_code != 200:
            print(f"❌ API 请求失败: {api_response.status_code}")
            return None
            
        json_data = api_response.json()
        print(f"✓ 获取到 {len(json_data)} 条记录")
        
        # 2. 转换为 DataFrame
        if isinstance(json_data, list) and len(json_data) > 0:
            df = pd.DataFrame(json_data)
            print(f"✓ 转换为 DataFrame，列数: {len(df.columns)}")
            print(f"列名: {list(df.columns)}")
            
            # 3. 转换为 CSV (Superset 更容易处理)
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()
            
            return {
                'dataframe': df,
                'csv_content': csv_content,
                'record_count': len(df),
                'columns': list(df.columns)
            }
        else:
            print("❌ JSON 数据格式不支持")
            return None
            
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        return None

def upload_csv_to_superset(csv_content, dataset_name):
    """上传 CSV 到 Superset"""
    
    # 1. 创建数据库连接 (如果需要)
    print(f"\n=== 检查数据库连接 ===")
    dbs_response = s.get(f"{SUPERSET_URL}/api/v1/database/")
    db_data = dbs_response.json()
    
    if db_data.get('count', 0) == 0:
        print("❌ 没有可用的数据库，请先在 Superset 中配置数据库")
        return None
    
    db_id = db_data['result'][0]['id']
    print(f"✓ 使用数据库 ID: {db_id}")
    
    # 2. 上传 CSV 文件
    print(f"\n=== 上传 CSV 数据集 ===")
    files = {
        'csv': (f'{dataset_name}.csv', csv_content, 'text/csv')
    }
    
    upload_response = s.post(
        f"{SUPERSET_URL}/api/v1/database/{db_id}/upload_csv/",
        files=files,
        data={
            'name': dataset_name,
            'description': f'从 JSON API 导入的数据集: {dataset_name}'
        }
    )
    
    if upload_response.status_code == 200:
        print(f"✓ 成功上传数据集: {dataset_name}")
        return upload_response.json()
    else:
        print(f"❌ 上传失败: {upload_response.status_code}")
        print(f"错误: {upload_response.text}")
        return None

# === 示例使用 ===
if __name__ == "__main__":
    
    # 示例 1: 使用公共 JSON API
    api_url = "https://jsonplaceholder.typicode.com/posts"
    dataset_info = create_json_dataset_from_api(api_url, "blog_posts")
    
    if dataset_info:
        print(f"\n📊 数据集信息:")
        print(f"  记录数: {dataset_info['record_count']}")
        print(f"  列数: {len(dataset_info['columns'])}")
        print(f"  列名: {dataset_info['columns']}")
        
        # 显示前几行数据
        print(f"\n前 3 行数据:")
        print(dataset_info['dataframe'].head(3).to_string())
        
        # 可选：上传到 Superset
        # result = upload_csv_to_superset(dataset_info['csv_content'], "blog_posts")
        # if result:
        #     print(f"✓ 数据集已上传到 Superset")
    
    print(f"\n💡 其他 JSON 数据源选项:")
    print(f"1. REST API 连接器 (需要配置)")
    print(f"2. 文件上传 (CSV/JSON)")
    print(f"3. 数据库导入")
    print(f"4. 自定义 Python 数据源")