from flask import Flask, render_template, request, jsonify, send_file, after_this_request
from flask_socketio import SocketIO
import threading
import os
import sqlite3
import time
import paramiko
import zipfile
import json
import re
import shutil
from urllib.parse import quote

app = Flask(__name__)
app.config['SECRET_KEY'] = 'web-fedprox-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

DB_PATH = '/home/ubuntu/project/WebFedProx/projects.db'
UPLOAD_PATH = '/home/ubuntu/project/WebFedProx/uploads'
os.makedirs(UPLOAD_PATH, exist_ok=True)

training_stop_flags = {}

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        description TEXT,
        n_clients INTEGER DEFAULT 2,
        server_ip TEXT,
        server_ssh_port INTEGER DEFAULT 22,
        server_user TEXT,
        server_pw TEXT,
        server_container TEXT,
        server_project_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS clients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER,
        rank INTEGER,
        ip TEXT,
        ssh_port INTEGER DEFAULT 22,
        user TEXT,
        pw TEXT,
        container TEXT,
        project_path TEXT,
        FOREIGN KEY (project_id) REFERENCES projects(id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS hyperparams (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER UNIQUE,
        epoch INTEGER DEFAULT 100,
        local_epoch INTEGER DEFAULT 5,
        timeout INTEGER DEFAULT 300,
        mu REAL DEFAULT 1.0,
        batch_size INTEGER DEFAULT 64,
        FOREIGN KEY (project_id) REFERENCES projects(id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER,
        round INTEGER,
        accuracy REAL,
        loss REAL,
        round_time REAL,
        participated TEXT,
        client_epochs TEXT,
        FOREIGN KEY (project_id) REFERENCES projects(id)
    )''')
    conn.commit()
    conn.close()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def ssh_connect(ip, port, user, pw):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ip, port=int(port), username=user, password=pw, timeout=10)
    return ssh

def sudo_exec(ssh, pw, command):
    stdin, stdout, stderr = ssh.exec_command(f"sudo -S {command}")
    stdin.write(pw + '\n')
    stdin.flush()
    stdout.channel.recv_exit_status()  # 원격 명령이 실제로 끝날 때까지 대기 (레이스 컨디션 방지)
    return stdout, stderr

def clean_output(text):
    lines = [l for l in text.split('\n')
             if not l.startswith('[sudo]') and 'password' not in l.lower()]
    return '\n'.join(lines).strip()

def get_docker_ports(ip, ssh_port, user, pw, container):
    try:
        ssh = ssh_connect(ip, ssh_port, user, pw)
        stdout, stderr = sudo_exec(ssh, pw,
            f"docker inspect {container} --format '{{{{json .HostConfig.PortBindings}}}}'"
        )
        result = clean_output(stdout.read().decode().strip())
        ssh.close()

        if not result or result == 'null' or result == '{}':
            return None, None

        port_data = json.loads(result)
        internal_port = None
        external_port = None

        for internal, bindings in port_data.items():
            if bindings:
                ext = bindings[0]['HostPort']
                inn = internal.split('/')[0]
                if inn not in ['22', '8080'] and ext not in ['22', '8080']:
                    internal_port = inn
                    external_port = ext
                    break

        return internal_port, external_port
    except Exception as e:
        socketio.emit('log', {'message': f'포트 읽기 오류: {str(e)}', 'type': 'error', 'project_id': 0})
        return None, None

def kill_existing_processes(ssh, pw, container):
    try:
        stdout, _ = sudo_exec(ssh, pw,
                              f"docker exec {container} pgrep -f 'main.py'"
        )
        pids = clean_output(stdout.read().decode()).strip()
        if pids:
            for pid in pids.split('\n'):
                pid = pid.strip()
                if pid:
                    sudo_exec(ssh, pw, f"docker exec {container} kill -9 {pid}")
            time.sleep(3)
            return True
        return False
    except:
        return False

def save_round_result(project_id, round_num, accuracy, loss, round_time, participated, client_epochs=None):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute('''INSERT INTO results (project_id, round, accuracy, loss, round_time, participated, client_epochs)
            VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (project_id, round_num, accuracy, loss, round_time, participated, client_epochs))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f'결과 저장 오류: {e}')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/terminal')
def terminal_page():
    return render_template('terminal.html')

@socketio.on('terminal_connect')
def handle_terminal_connect(data):
    import select
    ip = data['ip']
    port = data.get('ssh_port', 22)
    user = data['user']
    pw = data['pw']
    container = data['container']
    sid = request.sid

    if not hasattr(app, '_terminal_channels'):
        app._terminal_channels = {}

    def run_terminal():
        try:
            ssh = ssh_connect(ip, port, user, pw)
            channel = ssh.invoke_shell()
            channel.setblocking(0)
            channel.send(f"sudo docker exec -it {container} bash\n")
            time.sleep(1)
            channel.send(pw + '\n')
            time.sleep(0.5)
            app._terminal_channels[sid] = channel

            while True:
                r, _, _ = select.select([channel], [], [], 0.1)
                if r:
                    out = channel.recv(4096).decode('utf-8', errors='replace')
                    socketio.emit('terminal_output', {'data': out}, to=sid)
                if channel.closed:
                    break
        except Exception as e:
            socketio.emit('terminal_output', {'data': f'\r\n오류: {str(e)}\r\n'}, to=sid)

    t = threading.Thread(target=run_terminal, daemon=True)
    t.start()

@socketio.on('terminal_input')
def handle_terminal_input(data):
    sid = request.sid
    if hasattr(app, '_terminal_channels') and sid in app._terminal_channels:
        channel = app._terminal_channels[sid]
        if channel:
            channel.send(data['data'])

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    if hasattr(app, '_terminal_channels') and sid in app._terminal_channels:
        try:
            ch = app._terminal_channels[sid]
            if ch:
                ch.close()
        except:
            pass
        del app._terminal_channels[sid]

@app.route('/api/check_container', methods=['POST'])
def check_container():
    data = request.json
    try:
        ssh = ssh_connect(data['ip'], data.get('ssh_port', 22), data['user'], data['pw'])
        stdout, _ = sudo_exec(ssh, data['pw'],
            f"docker ps -a --format '{{{{.Names}}}}' | grep -w {data['container']}"
        )
        result = clean_output(stdout.read().decode()).strip()
        ssh.close()
        if result:
            return jsonify({'exists': True})
        else:
            return jsonify({'exists': False})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/list_images', methods=['POST'])
def list_images():
    data = request.json
    try:
        ssh = ssh_connect(data['ip'], data.get('ssh_port', 22), data['user'], data['pw'])
        stdout, _ = sudo_exec(ssh, data['pw'],
            "docker images --format '{{.ID}}|{{.Repository}}:{{.Tag}}'"
        )
        result = clean_output(stdout.read().decode()).strip()
        ssh.close()
        images = []
        for line in result.split('\n'):
            if not line.strip():
                continue
            img_id, name = line.split('|', 1)
            images.append({'id': img_id, 'name': name})
        return jsonify({'images': images})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/create_container', methods=['POST'])
def create_container():
    data = request.json
    try:
        ssh = ssh_connect(data['ip'], data.get('ssh_port', 22), data['user'], data['pw'])

        # ✅ 안전장치: 프론트에서 중복 체크를 하더라도, 서버에서 한 번 더 확인
        check_stdout, _ = sudo_exec(ssh, data['pw'],
            f"docker ps -a --format '{{{{.Names}}}}' | grep -w {data['container']}"
        )
        existing = clean_output(check_stdout.read().decode()).strip()
        if existing:
            ssh.close()
            return jsonify({'error': f"이미 존재하는 컨테이너 이름입니다: {data['container']}"}), 409

        cmd = (
            f"docker run -d --gpus all --name {data['container']} "
            f"-it {data['image']} bash"
        )
        stdout, _ = sudo_exec(ssh, data['pw'], cmd)
        result = clean_output(stdout.read().decode()).strip()
        ssh.close()
        return jsonify({'message': f"컨테이너 {data['container']} 생성 완료", 'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/projects', methods=['GET'])
def get_projects():
    conn = get_db()
    projects = conn.execute('SELECT * FROM projects ORDER BY created_at DESC').fetchall()
    conn.close()
    return jsonify([dict(p) for p in projects])

@app.route('/api/projects', methods=['POST'])
def create_project():
    data = request.json
    conn = get_db()
    c = conn.cursor()
    c.execute('''INSERT INTO projects
        (name, description, n_clients, server_ip, server_ssh_port,
         server_user, server_pw, server_container, server_project_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (data['name'], data.get('description', ''), data['n_clients'],
         data['server_ip'], data.get('server_ssh_port', 22),
         data['server_user'], data['server_pw'],
         data['server_container'], data['server_project_path']))
    project_id = c.lastrowid
    for i, client in enumerate(data.get('clients', [])):
        rank = client.get('rank', i)
        c.execute('''INSERT INTO clients
            (project_id, rank, ip, ssh_port, user, pw, container, project_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (project_id, rank, client['ip'], client.get('ssh_port', 22),
             client['user'], client['pw'],
             client['container'], client['project_path']))
    conn.commit()
    conn.close()
    return jsonify({'id': project_id, 'message': '프로젝트 생성 완료'})

@app.route('/api/projects/<int:project_id>', methods=['GET'])
def get_project(project_id):
    conn = get_db()
    project = conn.execute('SELECT * FROM projects WHERE id=?', (project_id,)).fetchone()
    clients = conn.execute('SELECT * FROM clients WHERE project_id=?', (project_id,)).fetchall()
    conn.close()
    if not project:
        return jsonify({'error': '프로젝트 없음'}), 404
    result = dict(project)
    result['clients'] = [dict(c) for c in clients]
    return jsonify(result)

@app.route('/api/projects/<int:project_id>', methods=['PUT'])
def update_project(project_id):
    data = request.json
    conn = get_db()
    conn.execute('''UPDATE projects SET
        name=?, description=?, n_clients=?, server_ip=?, server_ssh_port=?,
        server_user=?, server_pw=?, server_container=?, server_project_path=?
        WHERE id=?''',
        (data['name'], data.get('description', ''), data['n_clients'],
         data['server_ip'], data.get('server_ssh_port', 22),
         data['server_user'], data['server_pw'],
         data['server_container'], data['server_project_path'], project_id))
    conn.execute('DELETE FROM clients WHERE project_id=?', (project_id,))
    for i, client in enumerate(data.get('clients', [])):
        rank = client.get('rank', i)
        conn.execute('''INSERT INTO clients
            (project_id, rank, ip, ssh_port, user, pw, container, project_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (project_id, rank, client['ip'], client.get('ssh_port', 22),
             client['user'], client['pw'],
             client['container'], client['project_path']))
    conn.commit()
    conn.close()
    return jsonify({'message': '수정 완료'})

@app.route('/api/projects/<int:project_id>', methods=['DELETE'])
def delete_project(project_id):
    conn = get_db()
    conn.execute('DELETE FROM clients WHERE project_id=?', (project_id,))
    conn.execute('DELETE FROM hyperparams WHERE project_id=?', (project_id,))
    conn.execute('DELETE FROM results WHERE project_id=?', (project_id,))
    conn.execute('DELETE FROM projects WHERE id=?', (project_id,))
    conn.commit()
    conn.close()
    upload_path = os.path.join(UPLOAD_PATH, str(project_id))
    if os.path.exists(upload_path):
        shutil.rmtree(upload_path)
    return jsonify({'message': '삭제 완료'})

@app.route('/api/projects/<int:project_id>/hyperparams', methods=['POST'])
def save_hyperparams(project_id):
    data = request.json
    conn = get_db()
    conn.execute('''INSERT OR REPLACE INTO hyperparams
        (project_id, epoch, local_epoch, timeout, mu, batch_size)
        VALUES (?, ?, ?, ?, ?, ?)''',
        (project_id, data.get('epoch', 100), data.get('local_epoch', 5),
         data.get('timeout', 300), data.get('mu', 1.0), data.get('batch_size', 64)))
    conn.commit()
    conn.close()
    return jsonify({'message': '저장 완료'})

@app.route('/api/projects/<int:project_id>/hyperparams', methods=['GET'])
def get_hyperparams(project_id):
    conn = get_db()
    hp = conn.execute('SELECT * FROM hyperparams WHERE project_id=?', (project_id,)).fetchone()
    conn.close()
    if hp:
        return jsonify(dict(hp))
    return jsonify({'epoch':100,'local_epoch':5,'timeout':300,'mu':1.0,'batch_size':64})

@app.route('/api/projects/<int:project_id>/results', methods=['GET'])
def get_results(project_id):
    conn = get_db()
    results = conn.execute(
        'SELECT * FROM results WHERE project_id=? ORDER BY round ASC', (project_id,)
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in results])

@app.route('/api/projects/<int:project_id>/model/download', methods=['GET'])
def download_model(project_id):
    conn = get_db()
    project = conn.execute('SELECT * FROM projects WHERE id=?', (project_id,)).fetchone()
    conn.close()
    if not project:
        return jsonify({'error': '프로젝트 없음'}), 404
    project = dict(project)

    remote_tmp_path = f"/tmp/webfedprox_model_{project_id}.pkl"
    local_tmp_path = f"/tmp/webfedprox_model_download_{project_id}.pkl"

    try:
        ssh = ssh_connect(
            project['server_ip'], project['server_ssh_port'],
            project['server_user'], project['server_pw']
        )

        remote_model_path = f"{project['server_project_path']}/cache/global_model_state.pkl"

        # ✅ 컨테이너 안의 모델 파일을 원격 호스트로 꺼냄
        sudo_exec(ssh, project['server_pw'],
            f"docker cp {project['server_container']}:{remote_model_path} {remote_tmp_path}"
        )
        # sftp로 가져갈 수 있도록 권한 완화
        sudo_exec(ssh, project['server_pw'], f"chmod 644 {remote_tmp_path}")

        # 실제로 파일이 생겼는지 확인
        check_stdout, _ = sudo_exec(ssh, project['server_pw'], f"test -f {remote_tmp_path} && echo OK")
        if 'OK' not in clean_output(check_stdout.read().decode()):
            ssh.close()
            return jsonify({'error': '모델 파일을 찾을 수 없습니다. 학습이 최소 1라운드 이상 진행됐는지 확인해주세요.'}), 404

        # ✅ 원격 호스트 → WebFedProx 서버(로컬)로 다운로드
        sftp = ssh.open_sftp()
        sftp.get(remote_tmp_path, local_tmp_path)
        sftp.close()

        # 원격 호스트에 남긴 임시 파일 정리
        sudo_exec(ssh, project['server_pw'], f"rm -f {remote_tmp_path}")
        ssh.close()

        @after_this_request
        def cleanup(response):
            try:
                os.remove(local_tmp_path)
            except OSError:
                pass
            return response

        filename = f"{project['name']}_global_model.pkl"
        return send_file(local_tmp_path, as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/projects/<int:project_id>/results/download', methods=['GET'])
def download_results(project_id):
    from flask import Response
    conn = get_db()
    project = dict(conn.execute('SELECT * FROM projects WHERE id=?', (project_id,)).fetchone())
    results = conn.execute(
        'SELECT * FROM results WHERE project_id=? ORDER BY round ASC', (project_id,)
    ).fetchall()
    conn.close()

    def generate():
        n_clients = project['n_clients']
        client_headers = ','.join([f'client{i}' for i in range(n_clients)])
        yield f'round,accuracy,loss,{client_headers}\n'
        for r in results:
            accuracy = round(r['accuracy'] * 100, 2) if r['accuracy'] is not None else ''
            loss = round(r['loss'], 6) if r['loss'] is not None else ''
            client_epochs = r['client_epochs'].split(',') if r['client_epochs'] else []
            client_epochs += [''] * (n_clients - len(client_epochs))
            clients_str = ','.join([f'"=""{e}"""' if e else '""' for e in client_epochs])
            yield f"{r['round']},{accuracy},{loss},{clients_str}\n"

    filename = f"{project['name']}_results.csv"
    encoded_filename = quote(filename)
    return Response(generate(), mimetype='text/csv',
        headers={'Content-Disposition': f"attachment; filename=\"results.csv\"; filename*=UTF-8''{encoded_filename}"})

@app.route('/api/projects/<int:project_id>/results', methods=['DELETE'])
def clear_results(project_id):
    conn = get_db()
    conn.execute('DELETE FROM results WHERE project_id=?', (project_id,))
    conn.commit()
    conn.close()
    return jsonify({'message': '결과 초기화 완료'})

@app.route('/api/projects/<int:project_id>/upload', methods=['POST'])
def upload_file(project_id):
    f = request.files.get('file')
    rank = request.form.get('rank', 'server')
    if not f:
        return jsonify({'message': '파일 없음'}), 400

    if rank == 'server':
        upload_path = os.path.join(UPLOAD_PATH, str(project_id), 'server')
    else:
        upload_path = os.path.join(UPLOAD_PATH, str(project_id), f'client_{rank}')

    os.makedirs(upload_path, exist_ok=True)
    save_path = os.path.join(upload_path, f.filename)
    f.save(save_path)

    if f.filename.endswith('.zip'):
        with zipfile.ZipFile(save_path, 'r') as z:
            z.extractall(upload_path)

        # ✅ 업로드 완료 후 바로 도커로 전송
        conn = get_db()
        project = conn.execute('SELECT * FROM projects WHERE id=?', (project_id,)).fetchone()
        clients = [dict(c) for c in conn.execute('SELECT * FROM clients WHERE project_id=?', (project_id,)).fetchall()]
        conn.close()

        if project:
            project = dict(project)
            if rank == 'server':
                threading.Thread(
                    target=send_data_to_server,
                    args=(project, project_id),
                    daemon=True
                ).start()
            else:
                client_info = next((c for c in clients if str(c['rank']) == str(rank)), None)
                if client_info:
                    threading.Thread(
                        target=send_data_to_client,
                        args=(client_info, project_id),
                        daemon=True
                    ).start()

        return jsonify({'message': f'데이터 업로드 완료, 도커로 전송 시작 ({"서버" if rank == "server" else f"클라이언트 {rank}"})'})

    return jsonify({'message': f'{f.filename} 업로드 완료'})

def send_data_to_client(client_info, project_id):
    rank = client_info['rank']
    try:
        client_data_path = os.path.join(UPLOAD_PATH, str(project_id), f'client_{rank}')
        server_data_path = os.path.join(UPLOAD_PATH, str(project_id), 'server')

        if os.path.exists(client_data_path) and os.listdir(client_data_path):
            data_path = client_data_path
            label = f'클라이언트 {rank} 전용 데이터'
        elif os.path.exists(server_data_path) and os.listdir(server_data_path):
            data_path = server_data_path
            label = f'서버 공용 데이터'
        else:
            socketio.emit('log', {'message': f"[클라이언트 {rank}] 업로드된 데이터 없음 → 기존 데이터 사용", 'type': 'info', 'project_id': project_id})
            return

        ssh = ssh_connect(client_info['ip'], client_info['ssh_port'],
                         client_info['user'], client_info['pw'])

        remote_base = f"{client_info['project_path']}/dataset"
        sudo_exec(ssh, client_info['pw'],
            f"docker exec {client_info['container']} rm -rf {remote_base}"
        )
        time.sleep(0.5)
        sudo_exec(ssh, client_info['pw'],
            f"docker exec {client_info['container']} mkdir -p {remote_base}"
        )
        time.sleep(0.5)

        sftp = ssh.open_sftp()

        def upload_dir(local_dir, remote_dir):
            sudo_exec(ssh, client_info['pw'],
                f"docker exec {client_info['container']} mkdir -p {remote_dir}"
            )
            for item in os.listdir(local_dir):
                local_path = os.path.join(local_dir, item)
                remote_path = f"{remote_dir}/{item}"
                if os.path.isdir(local_path):
                    upload_dir(local_path, remote_path)
                else:
                    remote_tmp = f'/tmp/{item}'
                    sftp.put(local_path, remote_tmp)
                    sudo_exec(ssh, client_info['pw'],
                        f"docker cp {remote_tmp} {client_info['container']}:{remote_path}"
                    )
                    

        upload_dir(data_path, remote_base)
        sftp.close()
        ssh.close()
        socketio.emit('log', {'message': f"[클라이언트 {rank}] {label} 전송 완료", 'type': 'success', 'project_id': project_id})
        socketio.emit('transfer_complete', {'project_id': project_id, 'rank': rank, 'success': True})
    except Exception as e:
        socketio.emit('transfer_complete', {'project_id': project_id, 'rank': rank, 'success': False})
        socketio.emit('log', {'message': f"[클라이언트 {rank}] 전송 오류: {str(e)}", 'type': 'error', 'project_id': project_id})

def send_data_to_server(project_info, project_id):
    try:
        server_data_path = os.path.join(UPLOAD_PATH, str(project_id), 'server')
        if not os.path.exists(server_data_path) or not os.listdir(server_data_path):
            socketio.emit('log', {'message': '[서버] 업로드된 데이터 없음 → 기존 데이터 사용', 'type': 'info', 'project_id': project_id})
            return

        ssh = ssh_connect(project_info['server_ip'], project_info['server_ssh_port'],
                         project_info['server_user'], project_info['server_pw'])

        remote_base = f"{project_info['server_project_path']}/dataset"
        sudo_exec(ssh, project_info['server_pw'],
            f"docker exec {project_info['server_container']} rm -rf {remote_base}"
        )
        time.sleep(0.5)
        sudo_exec(ssh, project_info['server_pw'],
            f"docker exec {project_info['server_container']} mkdir -p {remote_base}"
        )
        time.sleep(0.5)

        sftp = ssh.open_sftp()

        def upload_dir(local_dir, remote_dir):
            sudo_exec(ssh, project_info['server_pw'],
                f"docker exec {project_info['server_container']} mkdir -p {remote_dir}"
            )
            for item in os.listdir(local_dir):
                local_path = os.path.join(local_dir, item)
                remote_path = f"{remote_dir}/{item}"
                if os.path.isdir(local_path):
                    upload_dir(local_path, remote_path)
                else:
                    remote_tmp = f'/tmp/{item}'
                    sftp.put(local_path, remote_tmp)
                    sudo_exec(ssh, project_info['server_pw'],
                        f"docker cp {remote_tmp} {project_info['server_container']}:{remote_path}"
                    )
                    

        upload_dir(server_data_path, remote_base)
        sftp.close()
        ssh.close()
        socketio.emit('transfer_complete', {'project_id': project_id, 'rank': 'server', 'success': True})
        socketio.emit('log', {'message': '[서버] 서버 공용 데이터 전송 완료', 'type': 'success', 'project_id': project_id})
    except Exception as e:
        socketio.emit('transfer_complete', {'project_id': project_id, 'rank': 'server', 'success': False})
        socketio.emit('log', {'message': f'[서버] 전송 오류: {str(e)}', 'type': 'error', 'project_id': project_id})

def run_training(project_id):
    time.sleep(2)
    conn = get_db()
    project = dict(conn.execute('SELECT * FROM projects WHERE id=?', (project_id,)).fetchone())
    clients = [dict(c) for c in conn.execute('SELECT * FROM clients WHERE project_id=?', (project_id,)).fetchall()]
    hp = dict(conn.execute('SELECT * FROM hyperparams WHERE project_id=?', (project_id,)).fetchone())
    conn.close()

    stop_flag = {'stop': False}
    training_stop_flags[project_id] = stop_flag

    def abort(message):
        stop_flag['stop'] = True
        training_stop_flags.pop(project_id, None)
        socketio.emit('log', {'message': f'❌ 학습 중단: {message}', 'type': 'error', 'project_id': project_id})
        socketio.emit('training_stopped', {'project_id': project_id})

    socketio.emit('log', {'message': '중앙 서버 포트 정보 읽는 중...', 'type': 'info', 'project_id': project_id})
    internal_port, external_port = get_docker_ports(
        project['server_ip'], project['server_ssh_port'],
        project['server_user'], project['server_pw'],
        project['server_container']
    )

    if not internal_port or not external_port:
        socketio.emit('log', {'message': '포트 자동 읽기 실패 → 기본값(8888/5000) 사용', 'type': 'error', 'project_id': project_id})
        internal_port = '8888'
        external_port = '5000'

    socketio.emit('log', {'message': f'포트 확인 → 서버 내부: {internal_port}, 클라이언트 접속: {external_port}', 'type': 'info', 'project_id': project_id})

    try:
        server_ssh = ssh_connect(
            project['server_ip'], project['server_ssh_port'],
            project['server_user'], project['server_pw']
        )

        killed = kill_existing_processes(server_ssh, project['server_pw'], project['server_container'])
        if killed:
            socketio.emit('log', {'message': '[서버] 기존 프로세스 종료 완료', 'type': 'info', 'project_id': project_id})
            time.sleep(10)

        sudo_exec(server_ssh, project['server_pw'],
            f"docker exec {project['server_container']} rm -f /tmp/server_log.txt"
        )
        time.sleep(0.5)

        server_cmd = (
            f"docker exec {project['server_container']} bash -c '"
            f"cd {project['server_project_path']} && "
            f"python3 -u main.py --role server "
            f"--port {internal_port} "
            f"--epoch {hp['epoch']} "
            f"--local_epoch {hp['local_epoch']} "
            f"--timeout {hp['timeout']} "
            f"--mu {hp['mu']} "
            f"--batch_size {hp['batch_size']} "
            f"--n_client {project['n_clients']} "
            f"--data_path {project['server_project_path']}/dataset/ "
            f"> /tmp/server_log.txt 2>&1 &'"
        )
        sudo_exec(server_ssh, project['server_pw'], server_cmd)
        socketio.emit('log', {'message': f'[서버] 학습 시작 (내부 포트: {internal_port})', 'type': 'info', 'project_id': project_id})
        time.sleep(2)

    except Exception as e:
        abort(f'[서버] 시작 오류: {str(e)}')
        return

    client_ssh_list = []
    for c in clients:
        try:
            ssh = ssh_connect(c['ip'], c['ssh_port'], c['user'], c['pw'])

            killed = kill_existing_processes(ssh, c['pw'], c['container'])
            if killed:
                socketio.emit('log', {'message': f"[클라이언트 {c['rank']}] 기존 프로세스 종료 완료", 'type': 'info', 'project_id': project_id})

            sudo_exec(ssh, c['pw'],
                f"docker exec {c['container']} rm -f /tmp/client_{c['rank']}_log.txt"
            )
            time.sleep(0.5)

            client_cmd = (
                f"docker exec {c['container']} bash -c '"
                f"cd {c['project_path']} && "
                f"python3 -u main.py --role client "
                f"--rank {c['rank']} "
                f"--server_ip {project['server_ip']} "
                f"--port {external_port} "
                f"--epoch {hp['epoch']} "
                f"--local_epoch {hp['local_epoch']} "
                f"--timeout {hp['timeout']} "
                f"--mu {hp['mu']} "
                f"--batch_size {hp['batch_size']} "
                f"--data_path {c['project_path']}/dataset/ "
                f"> /tmp/client_{c['rank']}_log.txt 2>&1 &'"
            )
            sudo_exec(ssh, c['pw'], client_cmd)
            socketio.emit('log', {'message': f"[클라이언트 {c['rank']}] 학습 시작 (접속 포트: {external_port})", 'type': 'info', 'project_id': project_id})
            client_ssh_list.append((ssh, c))
            time.sleep(1)
        except Exception as e:
            abort(f"[클라이언트 {c['rank']}] 오류: {str(e)}")
            return

    def monitor_logs():
        seen_server = 0
        seen_client = {c['rank']: 0 for c in clients}
        error_keywords = ['Traceback', 'RuntimeError', 'CUDNN_STATUS_NOT_INITIALIZED']
        current_round = {'num': 0, 'accuracy': None, 'loss': None, 'round_time': None, 'participated': None, 'client_epochs': []}

        while not stop_flag['stop']:
            try:
                stdout, _ = sudo_exec(server_ssh, project['server_pw'],
                    f"docker exec {project['server_container']} tail -n +{seen_server + 1} /tmp/server_log.txt 2>/dev/null"
                )
                all_lines = [l for l in stdout.read().decode().split('\n')
                             if l.strip() and not l.startswith('[sudo]') and 'password' not in l.lower()]

                for line in all_lines:
                    seen_server += 1
                    if '======' in line:
                        m = re.search(r'Round\s+(\d+)', line)
                        if m:
                            current_round['num'] = int(m.group(1))
                            current_round['accuracy'] = None
                            current_round['loss'] = None
                            current_round['round_time'] = None
                            current_round['participated'] = None
                            current_round['client_epochs'] = []
                        socketio.emit('log', {'message': line.strip(), 'type': 'round', 'project_id': project_id})
                    elif 'Avg Client Loss' in line:
                        socketio.emit('server_metric', {'message': line.strip(), 'project_id': project_id})
                        m = re.search(r'Loss: ([\d.]+)', line)
                        if m:
                            current_round['loss'] = float(m.group(1))
                        m2 = re.search(r'(\d+/\d+)', line)
                        if m2:
                            current_round['participated'] = m2.group(1)
                    elif '클라이언트 완료 epoch' in line:
                        m = re.search(r'(\d+/\d+)', line)
                        if m:
                            current_round['client_epochs'].append(m.group(1))
                    elif 'Test Accuracy' in line:
                        socketio.emit('server_metric', {'message': line.strip(), 'project_id': project_id})
                        socketio.emit('log', {'message': line.strip(), 'type': 'info', 'project_id': project_id})
                        m = re.search(r'([\d.]+)%', line)
                        if m:
                            current_round['accuracy'] = float(m.group(1)) / 100
                            if current_round['num'] > 0 and current_round['loss'] is not None:
                                client_epochs_str = ','.join(current_round['client_epochs']) if current_round['client_epochs'] else None
                                threading.Thread(
                                    target=save_round_result,
                                    args=(project_id, current_round['num'], current_round['accuracy'],
                                          current_round['loss'], current_round['round_time'],
                                          current_round['participated'], client_epochs_str),
                                    daemon=True
                                ).start()
                    elif '소요 시간' in line:
                        socketio.emit('server_metric', {'message': line.strip(), 'project_id': project_id})
                        m = re.search(r'([\d.]+)초', line)
                        if m:
                            current_round['round_time'] = float(m.group(1))
                    elif '참여 클라이언트' in line:
                        socketio.emit('server_metric', {'message': line.strip(), 'project_id': project_id})
                    elif '학습 종료' in line:
                        training_stop_flags.pop(project_id, None)
                        socketio.emit('log', {'message': '✅ 학습 완료!', 'type': 'success', 'project_id': project_id})
                        socketio.emit('training_stopped', {'project_id': project_id})
                        return
                    elif any(kw in line for kw in error_keywords):
                        abort(f'서버 오류 감지: {line.strip()}')
                        return

                for ssh, c in client_ssh_list:
                    try:
                        stdout, _ = sudo_exec(ssh, c['pw'],
                            f"docker exec {c['container']} tail -n +{seen_client[c['rank']] + 1} /tmp/client_{c['rank']}_log.txt 2>/dev/null"
                        )
                        c_lines = [l for l in stdout.read().decode().split('\n')
                                  if l.strip() and not l.startswith('[sudo]') and 'password' not in l.lower()]
                        for line in c_lines:
                            seen_client[c['rank']] += 1
                            socketio.emit('client_metric', {
                                'rank': c['rank'],
                                'message': line.strip(),
                                'project_id': project_id
                            })
                            if any(kw in line for kw in error_keywords):
                                abort(f'클라이언트 {c["rank"]} 오류 감지: {line.strip()}')
                                return
                    except:
                        pass

            except Exception as e:
                socketio.emit('log', {'message': f'모니터링 오류: {str(e)}', 'type': 'error', 'project_id': project_id})

            time.sleep(3)

    t = threading.Thread(target=monitor_logs)
    t.daemon = True
    t.start()

@app.route('/api/projects/<int:project_id>/train', methods=['POST'])
def start_training(project_id):
    thread = threading.Thread(target=run_training, args=(project_id,))
    thread.daemon = True
    thread.start()
    return jsonify({'message': '학습 시작'})

@app.route('/api/projects/<int:project_id>/stop', methods=['POST'])
def stop_training(project_id):
    conn = get_db()
    project = dict(conn.execute('SELECT * FROM projects WHERE id=?', (project_id,)).fetchone())
    clients = [dict(c) for c in conn.execute('SELECT * FROM clients WHERE project_id=?', (project_id,)).fetchall()]
    conn.close()

    if project_id in training_stop_flags:
        training_stop_flags[project_id]['stop'] = True
        training_stop_flags.pop(project_id, None)

    try:
        server_ssh = ssh_connect(
            project['server_ip'], project['server_ssh_port'],
            project['server_user'], project['server_pw']
        )
        kill_existing_processes(server_ssh, project['server_pw'], project['server_container'])
        server_ssh.close()
    except Exception as e:
        print(f'서버 종료 오류: {e}')

    for c in clients:
        try:
            ssh = ssh_connect(c['ip'], c['ssh_port'], c['user'], c['pw'])
            kill_existing_processes(ssh, c['pw'], c['container'])
            ssh.close()
        except Exception as e:
            print(f'클라이언트 {c["rank"]} 종료 오류: {e}')

    socketio.emit('log', {'message': '⏹ 학습이 강제 중단되었습니다.', 'type': 'error', 'project_id': project_id})
    socketio.emit('training_stopped', {'project_id': project_id})
    return jsonify({'message': '학습 중단'})

if __name__ == '__main__':
    init_db()
    socketio.run(app, host='0.0.0.0', port=8080, debug=False)
