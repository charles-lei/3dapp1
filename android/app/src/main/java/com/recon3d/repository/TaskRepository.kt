package com.recon3d.repository

import com.recon3d.network.TaskStatusDto
import kotlinx.coroutines.flow.Flow

class TaskRepository(private val dao: TaskDao) {
    val all: Flow<List<TaskEntity>> = dao.observeAll()

    fun byId(taskId: String): Flow<TaskEntity?> = dao.observeById(taskId)

    suspend fun insert(entity: TaskEntity) = dao.upsert(entity)

    /** 用服务端状态更新任务，保留本地字段（photoDir/glbPath/createdAt）。 */
    suspend fun upsertStatus(status: TaskStatusDto) {
        val existing = dao.getById(status.taskId)
        dao.upsert(
            TaskEntity(
                taskId = status.taskId,
                createdAt = existing?.createdAt ?: System.currentTimeMillis(),
                state = status.state,
                progress = status.progress,
                stageDetail = status.stageDetail,
                errorCode = status.error?.code,
                errorMessageZh = status.error?.messageZh,
                photoDir = existing?.photoDir,
                glbPath = existing?.glbPath,
            )
        )
    }

    suspend fun markGlb(taskId: String, path: String) {
        dao.getById(taskId)?.let { dao.upsert(it.copy(glbPath = path)) }
    }

    suspend fun delete(taskId: String) = dao.delete(taskId)
}
