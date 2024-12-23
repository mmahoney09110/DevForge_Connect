using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace DevForge_Connect.Migrations
{
    /// <inheritdoc />
    public partial class supportForNLP : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<string>(
                name: "AIGeneratedSummary",
                table: "ProjectSubmissions",
                type: "varchar(450)",
                nullable: false,
                defaultValue: "");

            migrationBuilder.AddColumn<string>(
                name: "NlpTags",
                table: "ProjectSubmissions",
                type: "varchar(450)",
                nullable: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "AIGeneratedSummary",
                table: "ProjectSubmissions");

            migrationBuilder.DropColumn(
                name: "NlpTags",
                table: "ProjectSubmissions");
        }
    }
}
