# See https://aka.ms/customizecontainer to learn how to customize your debug container and how Visual Studio uses this Dockerfile to build your images for faster debugging.

# This stage is used when running from VS in fast mode (Default for Debug configuration)
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS base
USER app
WORKDIR /app
EXPOSE 8080
EXPOSE 8081

# This stage is used to build the service project
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
ARG BUILD_CONFIGURATION=Release
WORKDIR /src
# Cài đặt các thư viện OpenCV cần thiết
RUN apt-get update && apt-get install -y libopencv-dev


COPY ["XacDinhHoaVan/XacDinhHoaVan.csproj", "XacDinhHoaVan/"]

# Khôi phục các dependencies mà không cần thêm nguồn nuget
RUN dotnet restore "./XacDinhHoaVan/XacDinhHoaVan.csproj"
COPY . .

# Build project
RUN dotnet build "./XacDinhHoaVan/XacDinhHoaVan.csproj" -c $BUILD_CONFIGURATION -o /app/build

# This stage is used to publish the service project to be copied to the final stage
FROM build AS publish
ARG BUILD_CONFIGURATION=Release
RUN dotnet publish "./XacDinhHoaVan/XacDinhHoaVan.csproj" -c $BUILD_CONFIGURATION -o /app/publish /p:UseAppHost=false

# This stage is used in production or when running from VS in regular mode (Default when not using the Debug configuration)
FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "XacDinhHoaVan.dll"]
