# 任务条目（任务层 /allocation/tasks 发布）
string task_id
string type              # SURVEY | EXCAVATE | TRANSPORT | DUMP
string assignee          # 指派机器人 id（空表示未指派）
int32 zone_id            # 关联资源区 id（-1 表示无）
string status            # PENDING | ASSIGNED | IN_PROGRESS | DONE | FAILED
