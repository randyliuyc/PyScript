import requests
import json

s = requests.Session()
login_payload = {"username": "randy", "password": "randy", "provider": "db"}

# 登录
login_response = s.post("http://124.71.144.80:9047/api/v1/security/login", json=login_payload)
print(f"✓ 登录成功 (状态码: {login_response.status_code})")

def get_resource_info(endpoint, name):
    """获取资源信息"""
    response = s.get(f"http://124.71.144.80:9047{endpoint}")
    if response.status_code == 200:
        data = response.json()
        count = data.get('count', 0)
        print(f"\n=== {name} (共 {count} 个) ===")
        
        if count > 0 and 'result' in data:
            items = data['result'][:5]  # 显示前5个
            for item in items:
                if name == "仪表板":
                    print(f"  - {item.get('dashboard_title', 'Unknown')}")
                elif name == "图表":
                    print(f"  - {item.get('slice_name', 'Unknown')}")
                elif name == "数据集":
                    print(f"  - {item.get('table_name', 'Unknown')}")
        
        return count
    else:
        print(f"❌ 获取{name}失败: {response.status_code}")
        return 0

# 获取各种资源信息
dashboard_count = get_resource_info("/api/v1/dashboard/", "仪表板")
chart_count = get_resource_info("/api/v1/chart/", "图表")
dataset_count = get_resource_info("/api/v1/dataset/", "数据集")

print(f"\n📊 Superset 资源概览:")
print(f"  • 仪表板: {dashboard_count} 个")
print(f"  • 图表: {chart_count} 个") 
print(f"  • 数据集: {dataset_count} 个")

# === 创建基于 JSON API 的数据集 ===
print(f"\n=== 创建 JSON API 数据集 ===")

# 示例：创建一个指向 JSON API 的数据集
dataset_payload = {
    "database": 1,  # 需要先有数据库连接
    "schema": "",
    "table_name": "json_api_data",
    "extra": json.dumps({
        "endpoint": "https://api.example.com/data",  # 你的 JSON API 端点
        "method": "GET",
        "headers": {
            "Content-Type": "application/json"
        }
    }),
    "description": "通过 JSON API 获取的数据集"
}

# 首先检查是否有可用的数据库
print("检查可用数据库...")
dbs_response = s.get("http://124.71.144.80:9047/api/v1/database/")
db_data = dbs_response.json()

if db_data.get('count', 0) == 0:
    print("❌ 没有可用的数据库连接")
    print("\n=== 创建示例 JSON 数据库连接 ===")
    
    # 创建一个指向 JSON 文件的数据库连接
    db_payload = {
        "database_name": "JSON_API_Database",
        " sqlalchemy_uri": "sqlite://",  # 使用 SQLite 作为示例
        "extra": json.dumps({
            "engine_params": {
                "connect_args": {
                    "check_same_thread": False
                }
            }
        }),
        "impersonate_user": False,
        "allow_ctas": True,
        "allow_cvas": True,
        "allow_dml": True,
        "allow_file_upload": True
    }
    
    create_db_response = s.post(
        "http://124.71.144.80:9047/api/v1/database/",
        json=db_payload
    )
    
    if create_db_response.status_code == 201:
        print("✓ 成功创建数据库连接")
        new_db = create_db_response.json()
        db_id = new_db.get('id')
        print(f"数据库 ID: {db_id}")
    else:
        print(f"❌ 创建数据库失败: {create_db_response.status_code}")
        print(f"错误信息: {create_db_response.text}")
else:
    print(f"✓ 找到 {db_data['count']} 个数据库连接")
    db_id = db_data['result'][0]['id']

# === 方法2：直接使用 REST API 连接器 ===
print(f"\n=== Superset 支持的 JSON 数据源方式 ===")
print("1. REST API 连接器 - 直接连接 JSON API")
print("2. 上传 JSON 文件 - 作为数据源")
print("3. 使用 Pandas API - 通过 Python 脚本处理 JSON")
print("4. 数据库表 - 存储 JSON 数据到数据库后连接")

# === 示例：获取外部 JSON API 数据并处理 ===
print(f"\n=== 示例：获取外部 JSON API 数据 ===")
try:
    # 示例：获取公共 JSON API
    api_response = requests.get("https://jsonplaceholder.typicode.com/posts", timeout=10)
    if api_response.status_code == 200:
        posts_data = api_response.json()
        print(f"✓ 成功获取 {len(posts_data)} 条 JSON 数据")
        print(f"示例数据: {json.dumps(posts_data[0], indent=2, ensure_ascii=False)}")
        
        # 可以将此数据保存为 CSV 或上传到 Superset
        print(f"\n💡 建议:")
        print(f"  1. 将 JSON 数据转换为 CSV 格式")
        print(f"  2. 上传 CSV 文件到 Superset")
        print(f"  3. 或创建数据库表存储这些数据")
        
    else:
        print(f"❌ 获取示例数据失败: {api_response.status_code}")
        
except Exception as e:
    print(f"❌ 网络请求失败: {e}")
    print("可以尝试其他 JSON API 或本地文件")