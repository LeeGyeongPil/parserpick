# 보안 정책 / Security policy

보안 취약점을 발견하셨다면 **공개 이슈 대신** 이 레포의 **Security → Report a vulnerability**(비공개 취약점 신고)로 알려 주세요. 확인 후 답변드리고, 고친 버전을 릴리스하면서 함께 안내합니다.

If you find a security issue, please **do not open a public issue**. Use **Security → Report a vulnerability** in this repository to report it privately. We will reply after checking and announce the fix with a release.

parserpick 은 포트를 열지 않는 CLI 이며, 받은 채팅 템플릿은 별도 프로세스의 Jinja 샌드박스에서만 렌더링하고 렌더링 시간 5초(출력 없는 루프는 7초 뒤 강제 종료)로 제한합니다. / parserpick is a CLI that opens no ports; chat templates it receives are rendered only inside a Jinja sandbox in a separate process that is limited to a 5-second rendering budget (a silent loop is force-killed at 7 seconds).
