import sqlite3
import pandas as pd
import sys
import ast

def find_post_id(image_path):
    # Connect to your SQLite database
    conn = sqlite3.connect("database.sqlite")  # change to your actual path if needed

    def find_post_id_by_image_path(image_path: str):
        query = """
        SELECT posts.id AS post_id
        FROM posts
        JOIN images ON posts.imageId = images.id
        WHERE images.path = ?
        """
        result = pd.read_sql_query(query, conn, params=(image_path,))
        return result

    post_info = find_post_id_by_image_path(image_path)

    # if not post_info.empty:
    post_id = post_info.iloc[0]['post_id']
    return post_id
        # print(f"Post ID for image '{image_path}': {post_id}")
    # else:
        # print(f"No post found for image '{image_path}'")

if __name__ == "__main__":
    # Example usage : python test_find_post_id.py '["image_10.png"]'
    image_paths = sys.argv[1]
    image_paths = ast.literal_eval(image_paths)
    print(find_post_id(image_paths[0]))