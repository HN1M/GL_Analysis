import os

def export_project_to_text(project_dir, output_file, exclude_dirs=None, exclude_files=None):
    """
    프로젝트 디렉토리 내의 모든 .py 파일 내용을 하나의 텍스트 파일로 합칩니다.

    Args:
        project_dir (str): 프로젝트 최상위 디렉토리 경로.
        output_file (str): 결과물이 저장될 텍스트 파일 이름.
        exclude_dirs (list, optional): 제외할 디렉토리 이름 리스트.
                                      기본값: ['venv', '.venv', 'env', '__pycache__', '.git'].
        exclude_files (list, optional): 제외할 파일 이름 리스트.
                                       기본값: [os.path.basename(__file__), output_file].
    """
    # 기본 제외 폴더 설정
    if exclude_dirs is None:
        exclude_dirs = ['venv', '.venv', 'env', '__pycache__', '.git']

    # 이 스크립트 파일 자체와 결과물 파일은 제외
    if exclude_files is None:
        exclude_files = [os.path.basename(__file__), output_file]

    print(f"'{output_file}' 파일 생성 시작...")

    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            # os.walk로 프로젝트 디렉토리 순회
            for root, dirs, files in os.walk(project_dir):
                # 제외할 디렉토리는 더 이상 탐색하지 않도록 설정
                dirs[:] = [d for d in dirs if d not in exclude_dirs]

                for filename in files:
                    # 파이썬 파일만 대상으로 하고, 제외 파일 리스트에 없어야 함
                    if filename.endswith('.py') and filename not in exclude_files:
                        file_path = os.path.join(root, filename)
                        # 프로젝트 루트 기준 상대 경로로 변환
                        relative_path = os.path.relpath(file_path, project_dir)
                        
                        print(f"  -> 추가 중: {relative_path}")

                        # 파일 경로를 구분선으로 추가
                        outfile.write(f"\n{'='*30}\n")
                        outfile.write(f"📄 FILE: {relative_path.replace(os.sep, '/')}\n")
                        outfile.write(f"{'='*30}\n\n")

                        try:
                            # 파일 내용 읽어서 쓰기
                            with open(file_path, 'r', encoding='utf-8') as infile:
                                outfile.write(infile.read())
                            outfile.write("\n\n")
                        except Exception as e:
                            outfile.write(f"--- 이 파일을 읽는 중 오류 발생: {e} ---\n\n")

        print(f"\n✅ 성공! 프로젝트 코드가 '{output_file}' 파일로 저장되었습니다.")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")


# --- 스크립트 실행 부분 ---
if __name__ == '__main__':
    # 1. 프로젝트 최상위 디렉토리 (현재 위치 '.'으로 설정)
    project_root_directory = '.'
    
    # 2. 결과가 저장될 파일 이름
    output_filename = 'project_code_export.txt'

    # 함수 실행
    export_project_to_text(project_root_directory, output_filename)