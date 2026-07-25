package com.recon3d.ui.task

import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.navigation.NavHostController

@Composable
fun TaskDetailScreen(nav: NavHostController, taskId: String) { Text("task $taskId") }
