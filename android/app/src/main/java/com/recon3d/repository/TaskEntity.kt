package com.recon3d.repository

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "tasks")
data class TaskEntity(
    @PrimaryKey val taskId: String,
    val createdAt: Long,
    val state: String,
    val progress: Int = 0,
    val stageDetail: String = "",
    val errorCode: String? = null,
    val errorMessageZh: String? = null,
    val photoDir: String? = null,
    val glbPath: String? = null,
)

private val ACTIVE_STATES = setOf("uploaded", "preprocessing", "colmap", "training", "mesh_export")

fun TaskEntity.isActive(): Boolean = state in ACTIVE_STATES

fun stateLabel(state: String): String = when (state) {
    "uploaded" -> "排队中"
    "preprocessing" -> "图像清洗"
    "colmap" -> "位姿估计"
    "training" -> "3DGS 训练"
    "mesh_export" -> "Mesh 导出"
    "done" -> "完成"
    "failed" -> "失败"
    else -> state
}
