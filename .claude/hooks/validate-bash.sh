#!/bin/bash
# validate-bash.sh — 위험한 Bash 명령어 차단 훅

COMMAND="$1"
DANGEROUS_PATTERNS=(
    "rm -rf"
    "sudo rm"
    "curl.*\|.*bash"
    "wget.*\|.*sh"
    "chmod 777"
    "sudo chmod"
    "> /dev/sda"
    "dd if=.*of=/dev"
    "mkfs\."
    "fdisk"
    ":(){:|:&};:"
)

for pattern in "${DANGEROUS_PATTERNS[@]}"; do
    if echo "$COMMAND" | grep -qE "$pattern"; then
        echo "🚫 차단: 위험한 명령어 감지 → $COMMAND"
        echo "   패턴 매칭: $pattern"
        echo "   이 명령어를 실행하려면 사용자가 직접 터미널에서 실행하세요."
        exit 1
    fi
done

exit 0
