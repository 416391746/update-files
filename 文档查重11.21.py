

"""
基于OOP模式开发，分为验证部分和主程序两部分
11.16 共有匹配逻辑修改为：全局完整出现：如果某个匹配内容在每个txt文件中都至少完整出现了一次，那么它应该被保存。
局部完整出现与全局子字符串出现：如果某个匹配内容至少在一个txt文件中完整出现，并且是其他每个txt文件中匹配内容的子字符串，那么它也应该被记录。
"""




import multiprocessing
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
import random
import json
import hashlib
from subprocess import check_output
import os
import logging
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from multiprocessing import Pool, Manager
from docx import Document
from queue import Queue, Empty
from threading import Thread
from collections import defaultdict



# 配置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class EmailVerifier:
    def __init__(self):
        self.sender_email = "416391746@qq.com"
        self.recipient_email = "416391746@qq.com"
        self.smtp_server = "smtp.qq.com"
        self.smtp_port = 465
        self.smtp_username = "416391746@qq.com"
        self.smtp_password = "edzlkttnmefibjeb"
        self.verify_status_file = "验证通过记录（误删）.json"

    def generate_code(self):
        return random.randint(100000, 999999)

    def send_verification_code(self, email, code):
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = email
            msg['Subject'] = "验证码"
            body = f"您的验证码为: {code}"
            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
            server.login(self.smtp_username, self.smtp_password)
            server.send_message(msg)
            server.quit()
            print(f"温馨提示：已发送验证码，请索要验证码！")
        except smtplib.SMTPAuthenticationError:
            print("认证错误，请检查用户名和密码是否正确。")
        except smtplib.SMTPServerDisconnected as e:
            print(f"连接意外关闭: {e}")
        except Exception as e:
            print(f"发送邮件时发生错误: {e}")

    def get_hard_disk_serial(self):
        try:
            output = check_output('wmic diskdrive get SerialNumber', shell=True, text=True)
            lines = output.split('\n')
            for line in lines:
                if line.strip().startswith('SerialNumber'):
                    return line.split()[-1].strip()
            return None
        except Exception as e:
            #print(f"获取硬盘序列号失败,无法正常验证: {e}")
            print(f"计算机信息更新失败,无法正常验证: {e}")
            return None

    def generate_hash_from_serial(self, serial):
        if serial:
            return hashlib.sha256(serial.encode()).hexdigest()
        return None

    def record_verification_status(self, verified=True, hard_disk_hash=None):
        try:
            with open(self.verify_status_file, 'w') as file:
                #data = {'verified': verified, 'hard_disk_hash': hard_disk_hash}
                data = {'verified': verified, 'secret': hard_disk_hash}
                json.dump(data, file)
        except Exception as e:
            print(f"写入验证状态文件时发生错误: {e}")

    def ask_for_code_and_verify(self, root):
        code = self.generate_code()
        self.send_verification_code(self.recipient_email, code)
        entered_code = simpledialog.askstring("验证码验证", "请输入您得到的验证码:")
        if entered_code == str(code):
            hard_disk_serial = self.get_hard_disk_serial()
            if hard_disk_serial:
                hard_disk_hash = self.generate_hash_from_serial(hard_disk_serial)
                self.record_verification_status(True, hard_disk_hash)
                #messagebox.showinfo("验证成功", "验证码和硬盘序列号验证成功！")
                messagebox.showinfo("验证成功", "信息验证成功！")
                self.send_email_notification(self.recipient_email, code, True)

                # 关闭当前窗口
                root.destroy()

                # 创建新的根窗口并启动主程序
                new_root = tk.Tk()
                new_root.title("文档查重工具")
                same_word_app = Same_word.DocumentHighlighterApp(new_root)
                new_root.mainloop()

            else:
                #messagebox.showerror("验证失败", "无法获取硬盘序列号，验证失败。")
                messagebox.showerror("验证失败", "无法更新计算机信息，验证失败。")
        else:
            messagebox.showerror("验证失败", "验证码错误，请重新验证。")
            self.send_email_notification(self.recipient_email, code, False)
            # 检查是否已经存在“重新发送验证码”按钮，如果存在则移除
            for widget in root.winfo_children():
                if isinstance(widget, tk.Button) and widget['text'] == "重新发送验证码":
                    widget.destroy()
                    break

            # 创建一个新的“重新发送验证码”按钮
            request_new_code_button = tk.Button(root, text="重新发送验证码",
                                                command=lambda: self.ask_for_code_and_verify(root))
            request_new_code_button.pack(pady=10)


    def send_email_notification(self, email, code, success):
        subject = "验证码验证完成" if success else "验证码验证失败"
        message = f"您的验证码为: {code}已验证成功" if success else f"未输入正确的验证码'{code}'导致验证失败"

        msg = MIMEMultipart()
        msg['From'] = self.sender_email
        msg['To'] = email
        msg['Subject'] = subject
        msg.attach(MIMEText(message, 'plain'))

        try:
            server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
            server.login(self.smtp_username, self.smtp_password)
            server.sendmail(self.sender_email, email, msg.as_string())
            server.quit()
            print("邮件发送成功。")
        except Exception as e:
            print(f"发送邮件时发生错误: {e}")

    def check_verification_status(self):
        try:
            if os.path.exists(self.verify_status_file):
                with open(self.verify_status_file, 'r') as file:
                    try:
                        status = json.load(file)
                        verified = status.get('verified', False)
                        #hard_disk_hash = status.get('hard_disk_hash')
                        hard_disk_hash = status.get('secret')
                        if verified and hard_disk_hash:
                            current_hard_disk_serial = self.get_hard_disk_serial()
                            if current_hard_disk_serial:
                                current_hard_disk_hash = self.generate_hash_from_serial(current_hard_disk_serial)
                                return current_hard_disk_hash == hard_disk_hash
                    except json.JSONDecodeError:
                        print("不能通过复制验证文件，请通过正规途径进行验证。")
                        return False
            return False
        except Exception as e:
            print(f"读取验证状态文件时发生错误: {e}")
            return False

#主程序
class Same_word:
    class DocumentComparer:
        def __init__(self, min_chars, file_paths, output_path):
            self.min_chars = min_chars
            self.file_paths = file_paths
            self.output_path = output_path
            self.documents = {}
            self.paragraph_positions = {}
            self.matches_all = {}
            self.common_matches = set()
            self.completed_comparisons = 0  # 添加为类实例变量
            self.interval_tree_root = None

        def load_documents(self):
            for fp in self.file_paths:
                try:
                    doc = Document(fp)
                    paragraphs = [p.text for p in doc.paragraphs]
                    positions = []
                    current_pos = 0
                    for paragraph in paragraphs:
                        end_pos = current_pos + len(paragraph)
                        positions.append((current_pos, end_pos))
                        current_pos = end_pos + 1
                    self.documents[fp] = paragraphs
                    self.paragraph_positions[fp] = positions
                    #logging.debug(f"Loaded document {fp}")
                    logging.debug(f"Loaded document ......")
                except Exception as e:
                    #logging.error(f"Failed to load document {fp}: {e}")
                    logging.error(f"Failed to load document ......")
                    raise

        def compare_two_docs(self, i, j):
            # 这个函数现在是类的一个方法，可以被pickle
            logging.debug(f"Comparing document {i} with document {j}")
            matches = []
            for k, paragraph1 in enumerate(self.documents[self.file_paths[i]]):
                for l, paragraph2 in enumerate(self.documents[self.file_paths[j]]):
                    paragraph_matches = self.sliding_window_compare(paragraph1, paragraph2, self.min_chars)
                    for match in paragraph_matches:
                        global_start1 = self.paragraph_positions[self.file_paths[i]][k][0] + match[0]
                        global_end1 = self.paragraph_positions[self.file_paths[i]][k][0] + match[1]
                        global_start2 = self.paragraph_positions[self.file_paths[j]][l][0] + match[2]
                        global_end2 = self.paragraph_positions[self.file_paths[j]][l][0] + match[3]
                        matches.append((global_start1, global_end1, global_start2, global_end2, match[4]))

            # 确保save_matches方法被调用,不管是不是匹配都要调用save_matches
            assert matches or self.save_matches([], os.path.basename(self.file_paths[i]), os.path.basename(
                self.file_paths[j])), "save_matches should be called when matches are empty"

            if matches:
                #logging.debug(f"Found {len(matches)} matches between {self.file_paths[i]} and {self.file_paths[j]}")
                logging.debug(f"Found *** matches between *** and ***")
                self.save_matches(matches, os.path.basename(self.file_paths[i]), os.path.basename(self.file_paths[j]))
            else:
                #logging.debug(f"No matches found between {self.file_paths[i]} and {self.file_paths[j]}")
                logging.debug(f"No matches found between *** and ***")
            return len(matches)

        def compare_documents(self, progress_queue):
            self.load_documents()
            total_comparisons = len(self.file_paths) * (len(self.file_paths) - 1) // 2
            completed_comparisons = 0

            with Pool() as pool:
                results = []
                for i in range(len(self.file_paths)):
                    for j in range(i + 1, len(self.file_paths)):
                        result = pool.apply_async(self.compare_two_docs, args=(i, j))
                        results.append(result)

                pool.close()
                pool.join()

                # 等待所有结果完成，并更新进度
                for _ in range(len(results)):
                    completed_comparisons += 1
                    progress = int(100 * completed_comparisons / total_comparisons)
                    progress_queue.put(progress)

                # 确保进度条更新到100%
                progress_queue.put(100)

        def sliding_window_compare(self, text1, text2, min_chars):
            matches = []
            len_text1 = len(text1)
            len_text2 = len(text2)
            logging.debug(f"Starting comparison ......")

            i = 0
            while i < len_text1 - min_chars + 1:
                for j in range(len_text2 - min_chars + 1):
                    if text1[i:i + min_chars] == text2[j:j + min_chars]:
                        end1, end2 = i + min_chars, j + min_chars
                        while end1 < len_text1 and end2 < len_text2 and text1[end1] == text2[end2]:
                            end1 += 1
                            end2 += 1
                        match_text = text1[i:end1]
                        matches.append((i, end1, j, end2, match_text))

                        # 检查是否可以合并前一个匹配
                        if matches and len(matches) > 1:
                            prev_match = matches[-2]
                            if prev_match[1] == i and prev_match[3] == j:
                                # 合并前一个匹配和当前匹配
                                matches[-2] = (prev_match[0], end1, prev_match[2], end2, text1[prev_match[0]:end1])
                                matches.pop()  # 移除当前匹配
                                i = end1  # 移动到合并后的匹配结束位置
                                break
                        else:
                            i = end1  # 移动到当前匹配的结束位置
                            break
                else:
                    i += 1
            logging.debug(f"comparison complete, found {len(matches)} matches.")
            return matches

        def update_matches_all(self, matches, doc_index1, doc_index2):
            for match in matches:
                match_text = match[4]
                start1 = match[0]
                start2 = match[2]
                if match_text not in self.matches_all:
                    self.matches_all[match_text] = {doc_index1: (start1, match[1]), doc_index2: (start2, match[3])}
                else:
                    self.matches_all[match_text][doc_index1] = (start1, match[1])
                    self.matches_all[match_text][doc_index2] = (start2, match[3])
                    # 检查是否有同一位置的更长匹配内容
                    for existing_match in list(self.matches_all.keys()):
                        if match_text in existing_match and match_text != existing_match:
                            existing_start_pos, existing_end_pos = self.matches_all[existing_match].get(doc_index1,
                                                                                                        (None, None))
                            if existing_start_pos == start1 and existing_end_pos == match[1]:
                                del self.matches_all[match_text]
                                break

        def save_matches(self, matches, file1_name, file2_name):
            # 去掉 .docx 后缀
            file1_name = os.path.splitext(file1_name)[0]  # 修改点：去掉 .docx 后缀
            file2_name = os.path.splitext(file2_name)[0]  # 修改点：去掉 .docx 后缀

            output_file = os.path.join(self.output_path, f"{file1_name}-{file2_name}[重复内容].txt")
            try:
                # 去重：确保同一位置的最长匹配内容优先
                unique_matches = []
                for match in matches:
                    if not any(
                            match[0] >= m[0] and match[1] <= m[1] and match[2] >= m[2] and match[3] <= m[3] for m in
                            unique_matches):
                        unique_matches.append(match)

                # 用于记录已经写入的匹配文本
                written_texts = set()



                with open(output_file, 'w', encoding='utf-8') as f:
                    for match in unique_matches:
                        match_text = match[4]
                        start1, end1, start2, end2 = match[:4]
                        if match_text not in written_texts:
                            f.write(f"{match_text}\n")
                            # f.write(f"Start position in {file1_name}: {start1}\n")
                            # f.write(f"End position in {file1_name}: {end1}\n")
                            # f.write(f"Start position in {file2_name}: {start2}\n")
                            # f.write(f"End position in {file2_name}: {end2}\n")
                            # f.write("-" * 40 + "\n")
                            written_texts.add(match_text)
                # 记录日志
                if matches:
                    logging.debug(f"Saved matches to {output_file}")
                else:
                    logging.debug(f"Created an empty matches file: {output_file}")

            except Exception as e:
                logging.error(f"Failed to save matches to {output_file}: {e}")




        def save_common_matches(self):
            # 获取所有生成的匹配文件
            match_files = [
                os.path.join(self.output_path,f"{os.path.splitext(os.path.basename(file1))[0]}-{os.path.splitext(os.path.basename(file2))[0]}[重复内容].txt")
                for i, file1 in enumerate(self.file_paths)
                for j, file2 in enumerate(self.file_paths)
                if i < j
            ]

            # 初始化字典来存储每个匹配内容的出现情况
            match_presence = {}
            for match_file in match_files:
                try:
                    if not os.path.exists(match_file):
                        continue
                    with open(match_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                if line not in match_presence:
                                    match_presence[line] = {'full_matches': 0, 'substring_matches': 0, 'files': set()}
                                match_presence[line]['files'].add(match_file)
                except Exception as e:
                    logging.error(f"Failed to read match file {match_file}: {e}")

            # 检查每个匹配内容是否全局完整出现或局部完整出现且为其他文件的子字符串
            for match_file in match_files:
                if not os.path.exists(match_file):
                    continue
                with open(match_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    for match, data in match_presence.items():
                        if match in content:
                            data['full_matches'] += 1
                        elif any(match in substring for substring in content.split()):
                            data['substring_matches'] += 1

            # 筛选出满足条件的匹配内容
            self.common_matches = set()
            for match, data in match_presence.items():
                # 条件1：全局完整出现
                if data['full_matches'] == len(match_files):
                    self.common_matches.add(match)
                # 条件2：局部完整出现与全局子字符串出现
                elif data['full_matches'] >= 1 and data['substring_matches'] == len(match_files) - data['full_matches']:
                    self.common_matches.add(match)

            # 保存共同匹配内容到文件
            self.save_common_matches_to_file()

        def save_common_matches_to_file(self):
            output_file = os.path.join(self.output_path, "所有文档的重复内容.txt")
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    for match_text in self.common_matches:
                        f.write(f"{match_text}\n")
                # 记录日志
                if self.common_matches:
                    logging.debug(f"Saved common matches to {output_file}")
                else:
                    logging.debug(f"Created an empty common matches file: {output_file}")
            except Exception as e:
                logging.error(f"Failed to save common matches to {output_file}: {e}")


        def save_filtered_matches(self):
            # 获取所有生成的匹配文件
            match_files = [
                os.path.join(self.output_path,f"{os.path.splitext(os.path.basename(file1))[0]}-{os.path.splitext(os.path.basename(file2))[0]}[重复内容].txt")
                for i, file1 in enumerate(self.file_paths)
                for j, file2 in enumerate(self.file_paths)
                if i < j
            ]

            # 遍历所有匹配文件，生成新的输出文件，剔除掉所有共同部分的内容
            for match_file in match_files:
                try:
                    with open(match_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    filtered_lines = [line for line in lines if line.strip() not in self.common_matches]
                    new_output_file = match_file.replace("[重复内容].txt", "[重复内容（去除所有文档的重复内容）].txt")
                    with open(new_output_file, 'w', encoding='utf-8') as f:
                        for line in filtered_lines:
                            f.write(line)
                    # 记录日志
                    if filtered_lines:
                        logging.debug(f"Saved filtered matches to {new_output_file}")
                    else:
                        logging.debug(f"Created an empty filtered matches file: {new_output_file}")
                except Exception as e:
                    logging.error(f"Failed to process match file {match_file}: {e}")



        def start_comparison(self, progress_queue):
            self.progress_queue = progress_queue
            self.compare_documents()

    class DocumentHighlighterApp:
        def __init__(self, root):
            self.root = root
            self.root.title("文档查重工具")
            # 文件列表
            self.file_paths = []
            # 输出路径
            self.output_path = ""  # 初始化output_path属性为空字符串
            # 显示信息的文本框
            self.info_text = tk.Text(root, wrap='word', height=10)
            self.info_text.pack(pady=10)
            # 文件选择按钮
            self.file_button = tk.Button(root, text="选择文件", command=self.select_files)
            self.file_button.pack(pady=10)
            # 清空文件列表按钮
            self.clear_files_button = tk.Button(root, text="清空文件列表", command=self.clear_files)
            self.clear_files_button.pack(pady=10)
            # 最小字符数输入框
            self.min_chars_label = tk.Label(root, text="最小字符数:")
            self.min_chars_label.pack()
            self.min_chars_entry = tk.Entry(root)
            self.min_chars_entry.insert(0, "5")  # 默认值
            self.min_chars_entry.pack()
            # 输出路径选择按钮
            self.output_button = tk.Button(root, text="选择输出路径", command=self.select_output_path)
            self.output_button.pack(pady=10)
            # 开始按钮
            self.start_button = tk.Button(root, text="开始比较", command=self.start_comparison)
            self.start_button.pack(pady=10)
            # 进度条
            self.progress_var = tk.IntVar()
            self.progress_bar = ttk.Progressbar(root, variable=self.progress_var, maximum=100)
            self.progress_bar.pack(pady=10)

            self.qq_label = ttk.Label(root, text="QQ：1011489402", foreground="blue", cursor="hand2")
            self.qq_label.pack( pady= 10)

        def select_files(self):
            new_file_paths = filedialog.askopenfilenames(filetypes=[("Word Documents", "*.docx")])
            if new_file_paths:  # 检查是否有文件被选择
                self.file_paths.extend(new_file_paths)  # 将新选择的文件添加到列表中
                self.info_text.insert(tk.END, f"选择了文件: {', '.join(new_file_paths)}\n")
            else:
                self.info_text.insert(tk.END, "未选择任何文件\n")

        def clear_files(self):
            self.file_paths = []  # 清空文件列表
            self.info_text.delete('1.0', tk.END)  # 清空文本框内容
            self.info_text.insert(tk.END, "文件列表已清空\n")

        def select_output_path(self):
            new_output_path = filedialog.askdirectory()
            if new_output_path:  # 检查是否有路径被选择
                self.output_path = new_output_path  # 更新输出路径
                self.info_text.insert(tk.END, f"输出路径设置为: {self.output_path}\n")
            else:
                self.info_text.insert(tk.END, "未选择输出路径，保持之前的输出路径\n")

        def start_comparison(self):
            if len(self.file_paths) < 2:
                messagebox.showwarning("警告", "请至少导入两个文件进行查重")
                return
            if not self.file_paths:
                messagebox.showwarning("警告", "请选择要比较的文件")
                return
            if not self.output_path:
                messagebox.showwarning("警告", "请选择输出路径")
                return
            try:
                min_chars = int(self.min_chars_entry.get())
                if min_chars < 1:
                    raise ValueError("最小字符数必须大于0")
            except ValueError as e:
                messagebox.showerror("错误", str(e))
                return
            self.progress_var.set(0)
            self.info_text.insert(tk.END, "开始比较.......\n请耐心等待......\n")
            self.info_text.see(tk.END)

            # 正确引用DocumentComparer类
            self.comparer = Same_word.DocumentComparer(min_chars, self.file_paths, self.output_path)
            self.progress_queue = Queue()
            self.thread = Thread(target=self.comparer.compare_documents, args=(self.progress_queue,))
            self.thread.start()
            self.check_progress()

        def check_progress(self):
            try:
                while True:
                    progress = self.progress_queue.get()
                    if progress == 100:
                        self.info_text.insert(tk.END, "比较完成\n")
                        self.info_text.see(tk.END)
                        self.comparer.save_common_matches()
                        self.comparer.save_filtered_matches()
                        break
                    self.progress_var.set(progress)
            except Empty:
                self.root.after(100, self.check_progress)
            except KeyboardInterrupt:
                print("程序被用户中断")
                # 这里可以添加任何需要的清理代码
                self.root.quit()  # 安全退出Tkinter主循环
                self.root.destroy()






def main():
    #由于使用了多线程，所以打包exe时出现了多模块不能导入，需要在这里加这条语句，还要在开始就导入multiprocessing
    multiprocessing.freeze_support()
    # 创建 EmailVerifier 实例
    email_verifier = EmailVerifier()
    # 创建 Tkinter 根窗口
    root = tk.Tk()
    root.title("应用程序")

    # 检查验证状态
    if not email_verifier.check_verification_status():
        # 如果未验证，提示用户进行验证
        email_verifier.ask_for_code_and_verify(root)
    else:
        # 如果已验证，启动 Same_word 主程序
        same_word_app = Same_word.DocumentHighlighterApp(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("程序被用户中断")
        # 这里可以添加任何需要的清理代码
        root.destroy()

if __name__ == "__main__":
    main()