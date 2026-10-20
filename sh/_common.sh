#!/usr/bin/env bash
#
# [개선 5] 실행 스크립트 공통 함수. run_agent_*.sh / run_frontend.sh가 source로 불러 쓴다.
#
# 원본 스크립트는 `lsof -ti:8100 | xargs kill -9`로 포트 주인이 누구든 강제 종료했다.
# 이 PC에서는 8100을 Docker Desktop 백엔드가 쓰고 있었고, 그대로 실행하면 Docker 전체가
# 멈출 뻔했다. 여기서는 "이 프로젝트의 서버(server.py / web_demo_server.py)일 때만" 끈다.

# free_port <포트> <우리 프로세스로 인정할 명령 패턴>
#   명령 이름이 맞고(server.py 등) + 그 프로세스의 작업 폴더(cwd)가 '이 프로젝트 폴더'일 때만 종료.
#   다른 폴더(예: 원본 all_total_vue_student)에서 띄운 server.py는 건드리지 않는다.
free_port() {
    local port="$1" pattern="$2" pid cmd cwd
    for pid in $(lsof -ti:"$port" -sTCP:LISTEN 2>/dev/null); do
        cmd="$(ps -o command= -p "$pid" 2>/dev/null || true)"
        cwd="$(lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p')"
        if echo "$cmd" | grep -Eq "$pattern" && [ "$cwd" = "$PWD" ]; then
            echo "이전에 띄운 서버를 종료합니다 (PID $pid: $cmd)"
            kill "$pid" 2>/dev/null || true          # 먼저 정상 종료 요청(SIGTERM)
            for _ in $(seq 1 10); do kill -0 "$pid" 2>/dev/null || break; sleep 0.3; done
            kill -9 "$pid" 2>/dev/null || true       # 그래도 안 죽었을 때만 강제 종료
        else
            echo "ERROR: 포트 $port 를 다른 프로그램이 쓰고 있습니다 → $cmd (작업 폴더: ${cwd:-알 수 없음})"
            echo "  그 프로그램을 끄거나, 다른 포트로 실행하세요. 예: BACKEND_PORT=8101 bash $0"
            exit 1
        fi
    done
}
