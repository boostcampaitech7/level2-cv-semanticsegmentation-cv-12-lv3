from notion_client import Client
from datetime import datetime
import pytz
import time

# txt 파일에서 불러오기
def read_notion_credentials():
    try:
        with open('/data/ephemeral/home/notion_credentials.txt', 'r') as f:
            auth_key = f.readline().strip()
            database_id = f.readline().strip()
        return auth_key, database_id
    except FileNotFoundError:
        raise FileNotFoundError("notion_credentials.txt 파일을 찾을 수 없습니다.")
    except Exception as e:
        raise Exception(f"Credentials 읽기 오류: {str(e)}")
    
def init_notion_client():
    try:
        auth_key, db_id = read_notion_credentials()
        return Client(auth=auth_key), db_id
    except Exception as e:
        print(f"Notion 클라이언트 초기화 실패: {str(e)}")
        raise

notion, database_id = init_notion_client()

# 페이지 내용 삭제
def clear_page_content(page_id):
    try:
        blocks = notion.blocks.children.list(block_id=page_id)

        for block in blocks["results"]:
            notion.blocks.delete(block_id=block["id"])
    except Exception as e:
        print(f"페이지 내용 삭제 에러: {str(e)}")

def insert_page_content(page_id, experiment_info=None, user_name=None):
    clear_page_content(page_id)

    time.sleep(1)

    # 실험 내용이나 사람 이름이 있으면
    if experiment_info or user_name:
        current_time = datetime.now(pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')
        children = [
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"type": "text", "text": {"content": "실험 정보"}}]
                }
            },
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": f"실험 내용: {experiment_info['description']}"}}]
                }
            },
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content" : f"실험 시작 시간: {current_time}"}}]
                }
            },
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content" : f"실험자: {user_name}"}}]
                }
            }
        ]

        notion.blocks.children.append(
            block_id=page_id,
            children=children
        )

def update_gpu_server_status(server_id, status, experiment_info=None, user_name=None):
    """
    GPU 상태 업데이트
    """
    kst = pytz.timezone('Asia/Seoul')
    current_time = datetime.now(kst).isoformat()

    try:
        query = notion.databases.query(
            database_id=database_id,
            filter = {
                "property": "이름",
                "title": {"equals": server_id}
            }
        )

        # 노션은 예민해서 바로바로 실행해버리면 충돌 에러 발생
        time.sleep(1)

        properties = {
            "이름": {
                "title": [
                    {
                        "type": "text",
                        "text": {
                            "content": server_id
                        }
                    }
                ]
            },
            "태그": {  # Status 속성 추가
                "multi_select": [
                    {"name": status}
                ]
            }
        }

        if query["results"]:
            page_id = query["results"][0]["id"]
            notion.pages.update(
                page_id=page_id, 
                properties=properties
            )

            time.sleep(1)

            if status == "RUN":
                insert_page_content(page_id, experiment_info, user_name)
            elif status == "IDLE":
                clear_page_content(page_id)
            print(f"{server_id} 상태가 {status}로 변경되었습니다.")
        else:
            response = notion.pages.create(
                parent={"database_id": database_id},
                properties=properties
            )

            time.sleep(1)

            if status == "RUN":
                insert_page_content(response["id"], experiment_info, user_name)
            print(f"{server_id} 새로운 항목이 생성되었습니다.")
    except Exception as e:
        print(f"상태 업데이트 중 에러 발생: {str(e)}")

def start_server(server_id, experiment_description, user_name):
    experiment_info = {
        "description": experiment_description
    }
    update_gpu_server_status(server_id, "RUN", experiment_info, user_name)

def stop_server(server_id):
    update_gpu_server_status(server_id, "IDLE")

def check_database_structure():
    try:
        database = notion.databases.retrieve(database_id=database_id)
        print("데이터베이스 속성:")
        for prop_name, prop_info in database['properties'].items():
            print(f"- {prop_name} ({prop_info['type']})")
    except Exception as e:
        print(f"데이터베이스 구조 확인 실패: {str(e)}")

