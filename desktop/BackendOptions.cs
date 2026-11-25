using System.ComponentModel;
using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Qolda;

internal sealed record BackendOptions
{
    private const string Engine = "../engine-win64/";
    private const string Runtime = Engine + "runtime/";
    public bool AttachOnly { get; init; }
    public string BackendRoot { get; init; } = Engine + "app";
    public string Python { get; init; } = Engine + "python/python.exe";
    public string Bundle { get; init; } = Runtime + "bundle.json";
    public string Artifacts { get; init; } = Runtime;
    public string PoseModel { get; init; } = Runtime + "end2end.onnx";
    public string Ffprobe { get; init; } = Runtime + "ffprobe.exe";
    public string Scratch { get; init; } = "%LOCALAPPDATA%/Qolda/scratch";
    public string? DllDir { get; init; }

    public static (BackendOptions Options, string BaseDir) Load(string? explicitConfig)
    {
        try
        {
            var exeDir = AppContext.BaseDirectory;
            var config = Path.GetFullPath(explicitConfig ?? "qolda.local.json", exeDir);
            RequireLocalDrive(config, "--config");
            if (explicitConfig is null && !File.Exists(config)) return (new(), exeDir);
            var options = JsonSerializer.Deserialize<BackendOptions>(File.ReadAllText(config), new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow
            }) ?? throw new JsonException();
            return (options, Path.GetDirectoryName(config)!);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or JsonException or ArgumentException or NotSupportedException)
        {
            throw new StartupException("Не удалось прочитать конфигурацию движка. Проверьте путь и формат JSON.");
        }
    }

    public Process Start(string baseDir)
    {
        var root = LocalPath(BackendRoot, baseDir, "backendRoot");
        var python = LocalPath(Python, baseDir, "python");
        var bundle = LocalPath(Bundle, baseDir, "bundle");
        var artifacts = LocalPath(Artifacts, baseDir, "artifacts");
        var pose = LocalPath(PoseModel, baseDir, "poseModel");
        var ffprobe = LocalPath(Ffprobe, baseDir, "ffprobe");
        var scratch = LocalPath(Scratch, baseDir, "scratch");
        var dllDir = DllDir is null ? null : LocalPath(DllDir, baseDir, "dllDir");
        var missing = new List<string>();
        if (!File.Exists(python)) missing.Add("python.exe");
        if (!File.Exists(Path.Combine(root, "backend", "api.py")) ||
            !File.Exists(Path.Combine(root, "backend", "runtime.py")) ||
            !File.Exists(Path.Combine(root, "ml", "gpu.py"))) missing.Add("backendRoot (backend/ и ml/)");
        if (!File.Exists(Path.Combine(root, "frontend", "index.html"))) missing.Add("frontend/index.html");
        if (!File.Exists(bundle)) missing.Add("bundle.json");
        if (!File.Exists(Path.Combine(Path.GetDirectoryName(bundle)!, "end2end.b1.onnx"))) missing.Add("end2end.b1.onnx рядом с bundle");
        if (!File.Exists(Path.Combine(artifacts, "A-tcn.safetensors"))) missing.Add("A-tcn.safetensors");
        if (!File.Exists(pose)) missing.Add("end2end.onnx");
        if (!File.Exists(ffprobe)) missing.Add("ffprobe.exe");
        if (dllDir is not null && !Directory.Exists(dllDir)) missing.Add("dllDir");
        if (missing.Count > 0)
            throw new StartupException("Движок не установлен: " + string.Join(", ", missing) +
                ". Поместите engine-win64 рядом с папкой Qolda или исправьте qolda.local.json.");

        var relativeScratch = Path.GetRelativePath(root, scratch);
        if (!Path.IsPathRooted(relativeScratch) && relativeScratch != ".." &&
            !relativeScratch.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal))
            throw new StartupException("Каталог scratch должен быть вне backendRoot и вне Git.");
        try { Directory.CreateDirectory(scratch); }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            throw new StartupException("Нет доступа к каталогу scratch для временных видео.");
        }

        var start = new ProcessStartInfo(python)
        {
            WorkingDirectory = root,
            UseShellExecute = false,
            CreateNoWindow = true
        };
        string[] arguments = ["-B", "-E", "-m", "backend.api", "--port", LocalEndpoint.DefaultPort, "--bundle", bundle,
            "--artifacts", artifacts, "--pose-model", pose, "--private-scratch", scratch, "--ffprobe", ffprobe];
        foreach (var argument in arguments) start.ArgumentList.Add(argument);
        if (dllDir is not null)
        {
            start.ArgumentList.Add("--dll-dir");
            start.ArgumentList.Add(dllDir);
        }
        try { return Process.Start(start) ?? throw new StartupException("Не удалось запустить Python движка."); }
        catch (Win32Exception) { throw new StartupException("Не удалось запустить Python движка. Проверьте версию, права и зависимости."); }
    }

    private static string LocalPath(string? value, string baseDir, string field)
    {
        if (string.IsNullOrWhiteSpace(value)) throw new StartupException($"В конфигурации не задано поле {field}.");
        try
        {
            var full = Path.GetFullPath(Environment.ExpandEnvironmentVariables(value), baseDir);
            RequireLocalDrive(full, field);
            return full;
        }
        catch (Exception e) when (e is ArgumentException or NotSupportedException or IOException or UnauthorizedAccessException)
        {
            throw new StartupException($"Некорректный путь в поле {field}.");
        }
    }

    private static void RequireLocalDrive(string path, string field)
    {
        var drive = Path.GetPathRoot(path);
        if (drive is null || path.StartsWith(@"\\", StringComparison.Ordinal) ||
            new DriveInfo(drive).DriveType == DriveType.Network)
            throw new StartupException($"Поле {field} должно указывать на локальный диск.");
    }
}

internal sealed class StartupException(string message) : Exception(message);
